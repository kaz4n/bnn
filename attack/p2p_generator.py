#!/usr/bin/env python3
"""Power2Picture-style generative input recovery for the CW305 leakage core.

Follows Huegle et al., "Power2Picture" (FCCM 2023): instead of decoding pixels
with a hand-built power template, a generative CNN is trained by regression to
map a measured power trace directly to the input image.

Generator (their Table I):
    Linear 128 -> Linear 128 -> Linear 12544 -> reshape (256,7,7)
    -> ConvT 128 (5,5) s1 p2 -> ConvT 64 (4,4) s2 p1 -> ConvT 1 (4,4) s2 p1

Loss (their Eq. 1):
    gradmse = MSE(g,o) + MSE(dg/dx, do/dx) + MSE(dg/dy, do/dy)

Optimizer: Adam, lr 1e-3, batch 256.

Data comes from `host/cw305_p2p_capture.py`: one single-shot trace per image,
captured with the trained convolution kernel (no one-hot probe kernels), which
is what makes this attack model closer to the remote/profiled setting of the
paper than the template route.
"""
import argparse
import glob
import json
import os
import time

import numpy as np

RESULT_KIND = "cw305_power2picture_generator"


# ---------------------------------------------------------------- data


def load_shards(trace_dir, limit=None):
    manifest = json.load(open(os.path.join(trace_dir, "capture_manifest.json")))
    traces, images, labels, indices = [], [], [], []
    total = 0
    for path in sorted(glob.glob(os.path.join(trace_dir, "shard*.npz"))):
        d = np.load(path)
        traces.append(d["traces"])
        images.append(d["images"])
        labels.append(d["labels"])
        indices.append(d["indices"])
        total += len(d["traces"])
        if limit is not None and total >= limit:
            break
    traces = np.concatenate(traces)[:limit]
    images = np.concatenate(images)[:limit]
    labels = np.concatenate(labels)[:limit]
    indices = np.concatenate(indices)[:limit]
    scale = float(manifest.get("trace_scale", 32767.0))
    return traces.astype(np.float32) / scale, images, labels, indices, manifest


def featurize(traces, manifest, mode):
    """Reduce raw ADC samples to the generator input vector."""
    spc = int(manifest["samples_per_cycle"])
    dwell = int(manifest["dwell"])
    useful = int(manifest["useful_samples"])
    x = traces[:, :useful]
    x = x - np.median(x, axis=1, keepdims=True)
    if mode == "raw":
        return x
    if mode == "clock_abs":
        n = useful // spc
        return np.abs(x[:, :n * spc]).reshape(len(x), n, spc).mean(axis=2)
    if mode == "window_abs":
        spw = dwell * spc
        n = useful // spw
        return np.abs(x[:, :n * spw]).reshape(len(x), n, spw).mean(axis=2)
    raise ValueError(f"unknown feature mode {mode}")


# ---------------------------------------------------------------- metrics


def ssim_map(a, b, data_range=1.0, win=11):
    """Mean structural similarity, uniform 11x11 window (as in Wang et al.)."""
    from scipy.ndimage import uniform_filter

    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    mu_a = uniform_filter(a, win)
    mu_b = uniform_filter(b, win)
    saa = uniform_filter(a * a, win) - mu_a * mu_a
    sbb = uniform_filter(b * b, win) - mu_b * mu_b
    sab = uniform_filter(a * b, win) - mu_a * mu_b
    num = (2 * mu_a * mu_b + c1) * (2 * sab + c2)
    den = (mu_a ** 2 + mu_b ** 2 + c1) * (saa + sbb + c2)
    return float(np.mean(num / den))


def image_metrics(recovered01, truth01):
    """recovered01/truth01: (N,28,28) float in [0,1]; recovered is thresholded."""
    rec_bin = (recovered01 >= 0.5).astype(np.uint8)
    truth_bin = truth01.astype(np.uint8)
    tp = float(np.sum((rec_bin == 1) & (truth_bin == 1)))
    fp = float(np.sum((rec_bin == 1) & (truth_bin == 0)))
    fn = float(np.sum((rec_bin == 0) & (truth_bin == 1)))
    prec = tp / max(tp + fp, 1.0)
    rec = tp / max(tp + fn, 1.0)
    # Wei et al. pixel-level distance on the 0..255 scale
    pixel_dist = float(np.mean(np.abs(recovered01 * 255.0 - truth01 * 255.0)))
    mssim = float(np.mean([ssim_map(r, t) for r, t in zip(recovered01, truth01)]))
    return {
        "bit_acc": float(np.mean(rec_bin == truth_bin)),
        "foreground_precision": prec,
        "foreground_recall": rec,
        "foreground_f1": 2 * prec * rec / max(prec + rec, 1e-9),
        "foreground_iou": tp / max(tp + fp + fn, 1.0),
        "all_zero_bit_acc": float(np.mean(truth_bin == 0)),
        "mssim": mssim,
        "pixel_level_distance": pixel_dist,
    }


# ---------------------------------------------------------------- model


def build_model(in_dim, torch, nn):
    class Generator(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Sequential(
                nn.Linear(in_dim, 128), nn.BatchNorm1d(128), nn.LeakyReLU(0.2),
                nn.Linear(128, 128), nn.BatchNorm1d(128), nn.LeakyReLU(0.2),
                nn.Linear(128, 12544), nn.BatchNorm1d(12544), nn.LeakyReLU(0.2),
            )
            self.deconv = nn.Sequential(
                nn.ConvTranspose2d(256, 128, 5, stride=1, padding=2),
                nn.BatchNorm2d(128), nn.LeakyReLU(0.2),
                nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
                nn.BatchNorm2d(64), nn.LeakyReLU(0.2),
                nn.ConvTranspose2d(64, 1, 4, stride=2, padding=1),
            )

        def forward(self, x):
            x = self.fc(x).view(-1, 256, 7, 7)
            return torch.sigmoid(self.deconv(x)).squeeze(1)

    return Generator()


def gradmse(pred, target, torch, mse):
    loss = mse(pred, target)
    loss = loss + mse(pred[:, :, 1:] - pred[:, :, :-1],
                      target[:, :, 1:] - target[:, :, :-1])
    loss = loss + mse(pred[:, 1:, :] - pred[:, :-1, :],
                      target[:, 1:, :] - target[:, :-1, :])
    return loss


# ---------------------------------------------------------------- run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", default="../host/traces_p2p_9k")
    ap.add_argument("--out", default="results_p2p")
    ap.add_argument("--features", choices=["raw", "clock_abs", "window_abs"],
                    default="clock_abs")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--val-count", type=int, default=500)
    ap.add_argument("--test-count", type=int, default=500)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--loss", choices=["gradmse", "mse"], default="gradmse")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=0)
    args = ap.parse_args()

    import torch
    import torch.nn as nn

    if args.threads:
        torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.out, exist_ok=True)

    traces, images, labels, indices, manifest = load_shards(args.traces, args.limit)
    feats = featurize(traces, manifest, args.features)
    targets = images.astype(np.float32)
    n = len(feats)
    n_test = args.test_count
    n_val = args.val_count
    n_train = n - n_val - n_test
    if n_train <= 0:
        raise SystemExit("not enough captured images for the requested split")

    tr = slice(0, n_train)
    va = slice(n_train, n_train + n_val)
    te = slice(n_train + n_val, n)

    mu = feats[tr].mean(axis=0)
    sd = feats[tr].std(axis=0) + 1e-9
    feats = (feats - mu) / sd

    dev = torch.device("cpu")
    xt = torch.from_numpy(feats[tr]).float()
    yt = torch.from_numpy(targets[tr]).float()
    xv = torch.from_numpy(feats[va]).float()
    yv = torch.from_numpy(targets[va]).float()
    xe = torch.from_numpy(feats[te]).float()

    model = build_model(feats.shape[1], torch, nn).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    mse = nn.MSELoss()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"train={n_train} val={n_val} test={n_test} in_dim={feats.shape[1]} "
          f"params={n_params/1e6:.2f}M feature={args.features}", flush=True)

    best_val = float("inf")
    best_state = None
    history = []
    t0 = time.time()
    for epoch in range(args.epochs):
        model.train()
        perm = torch.randperm(n_train)
        total = 0.0
        for i in range(0, n_train, args.batch):
            idx = perm[i:i + args.batch]
            if len(idx) < 2:
                continue
            opt.zero_grad()
            pred = model(xt[idx])
            loss = (gradmse(pred, yt[idx], torch, mse) if args.loss == "gradmse"
                    else mse(pred, yt[idx]))
            loss.backward()
            opt.step()
            total += float(loss) * len(idx)
        model.eval()
        with torch.no_grad():
            vpred = model(xv)
            vloss = float(mse(vpred, yv))
            vacc = float(((vpred >= 0.5).float() == yv).float().mean())
        history.append({"epoch": epoch, "train_loss": total / n_train,
                        "val_mse": vloss, "val_bit_acc": vacc})
        if vloss < best_val:
            best_val = vloss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            # checkpoint immediately: long CPU runs should survive an interruption
            torch.save({"state_dict": best_state, "mu": mu, "sd": sd,
                        "features": args.features, "in_dim": int(feats.shape[1]),
                        "epoch": epoch, "val_mse": vloss},
                       os.path.join(args.out, "generator_best.pt"))
        print(f"epoch {epoch:3d} train={total/n_train:.5f} val_mse={vloss:.5f} "
              f"val_bit_acc={vacc:.4f} t={time.time()-t0:.0f}s", flush=True)

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        rec = model(xe).numpy()
    truth = targets[te]

    m = image_metrics(rec, truth)
    try:
        from recognize_numpy import GoldenMLP

        clf = GoldenMLP()
        lab = labels[te]
        pred_orig = clf.predict(truth)
        pred_rec = clf.predict((rec >= 0.5).astype(np.float32))
        pred_rec_gray = clf.predict(np.clip(rec, 0, 1))
        m["recognition_accuracy_original"] = float(np.mean(pred_orig == lab))
        m["recognition_accuracy_recovered"] = float(np.mean(pred_rec == lab))
        m["recognition_accuracy_recovered_gray"] = float(np.mean(pred_rec_gray == lab))
    except Exception as exc:  # recognition is optional
        m["recognition_error"] = str(exc)

    np.savez_compressed(os.path.join(args.out, "test_recovered.npz"),
                        recovered=rec.astype(np.float32),
                        truth=truth.astype(np.uint8),
                        labels=labels[te],
                        indices=indices[te])
    torch.save({"state_dict": best_state, "mu": mu, "sd": sd,
                "features": args.features, "in_dim": int(feats.shape[1])},
               os.path.join(args.out, "generator.pt"))
    payload = {
        "kind": RESULT_KIND,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "trace_dir": os.path.abspath(args.traces),
        "capture_manifest": manifest,
        "feature_mode": args.features,
        "loss": args.loss,
        "epochs": args.epochs,
        "batch": args.batch,
        "lr": args.lr,
        "n_train": n_train,
        "n_val": n_val,
        "n_test": n_test,
        "params": int(n_params),
        "metrics": m,
        "history": history,
    }
    with open(os.path.join(args.out, "summary.json"), "w") as fp:
        json.dump(payload, fp, indent=2)
    print(json.dumps(m, indent=2))


if __name__ == "__main__":
    main()
