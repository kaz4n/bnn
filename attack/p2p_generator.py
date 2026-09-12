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


def load_shards(trace_dir, limit=None, feature_mode=None, mmap_path=None):
    """Load a capture, optionally reducing each shard to features as it is read.

    The raw array is the memory bottleneck: 60000 traces of 22144 int16 samples
    is 4.95 GiB once promoted to float32, which does not fit alongside training.
    Passing feature_mode applies featurize() per shard and keeps only the reduced
    vectors, which for clock_abs is samples_per_cycle times smaller. Raw traces
    are still returned unreduced when feature_mode is None, so callers that need
    them are unaffected.
    """
    manifest = json.load(open(os.path.join(trace_dir, "capture_manifest.json")))
    scale = float(manifest.get("trace_scale", 32767.0))
    paths = sorted(glob.glob(os.path.join(trace_dir, "shard*.npz")))
    if not paths:
        raise SystemExit(f"no shard*.npz files found in {trace_dir}")

    # Stream into a preallocated array rather than building a list and calling
    # np.concatenate. At 60000 x 5408 float32 the result is 1.21 GiB and
    # concatenate needs the per-shard list alive alongside its output, so peak is
    # 2.4 GiB and it fails on this machine. Filling in place holds one shard extra.
    out = None
    images, labels, indices = [], [], []
    _mm = mmap_path  # when set, the feature array lives on disk, not in RAM
    total = 0
    for path in paths:
        d = np.load(path)
        t = d["traces"].astype(np.float32) / scale
        if feature_mode is not None:
            t = featurize(t, manifest, feature_mode)
        if out is None:
            n_total = int(manifest.get("n_images") or 0)
            if limit is not None:
                n_total = min(n_total, limit) if n_total else limit
            if not n_total:
                n_total = len(t) * len(paths)
            shape = (n_total,) + t.shape[1:]
            if _mm:
                # 60000 x 5408 float32 is 1.21 GiB and the caller then needs a
                # training-set copy of nearly the same size. Backing this by a file
                # keeps only the training tensor resident. Same dtype, same values.
                from numpy.lib.format import open_memmap
                out = open_memmap(_mm, mode="w+", dtype=t.dtype, shape=shape)
            else:
                out = np.empty(shape, dtype=t.dtype)
        take = len(t) if limit is None else min(len(t), len(out) - total)
        out[total:total + take] = t[:take]
        images.append(d["images"][:take])
        labels.append(d["labels"][:take])
        indices.append(d["indices"][:take])
        total += take
        del t, d
        if total >= len(out):
            break
    traces = out[:total]
    images = np.concatenate(images)[:total]
    labels = np.concatenate(labels)[:total]
    indices = np.concatenate(indices)[:total]
    return traces, images, labels, indices, manifest


def load_shards_permuted(trace_dir, feature_mode, order, limit=None):
    """Load a capture with rows placed at their split positions.

    The default path keeps the full feature array alive while fancy-indexing
    feats[tr], feats[va], feats[te] out of it, so peak memory is roughly twice the
    data. Writing row `order[i]` into position `i` at load time makes the three
    splits contiguous slices, which torch.from_numpy can wrap as views rather than
    copies. Same values, same order, one copy of the data.
    """
    manifest = json.load(open(os.path.join(trace_dir, "capture_manifest.json")))
    scale = float(manifest.get("trace_scale", 32767.0))
    paths = sorted(glob.glob(os.path.join(trace_dir, "shard*.npz")))
    if not paths:
        raise SystemExit(f"no shard*.npz files found in {trace_dir}")
    n = len(order)
    inv = np.empty(n, dtype=np.int64)
    inv[order] = np.arange(n, dtype=np.int64)

    feats = None
    images = labels = indices = None
    base = 0
    for path in paths:
        d = np.load(path)
        t = d["traces"].astype(np.float32) / scale
        if feature_mode is not None:
            t = featurize(t, manifest, feature_mode)
        take = min(len(t), n - base)
        if take <= 0:
            break
        dst = inv[base:base + take]
        if feats is None:
            feats = np.empty((n,) + t.shape[1:], dtype=t.dtype)
            im0 = d["images"]
            images = np.empty((n,) + im0.shape[1:], dtype=im0.dtype)
            labels = np.empty(n, dtype=d["labels"].dtype)
            indices = np.empty(n, dtype=d["indices"].dtype)
        feats[dst] = t[:take]
        images[dst] = d["images"][:take]
        labels[dst] = d["labels"][:take]
        indices[dst] = d["indices"][:take]
        base += take
        del t, d
    if base != n:
        raise SystemExit(f"expected {n} traces, shards supplied {base}")
    return feats, images, labels, indices, manifest


def assert_hardware_manifest(manifest):
    source = str(manifest.get("trace_source", "")).lower()
    mode = str(manifest.get("capture_mode", "")).lower()
    if "sim" in source or "sim" in mode:
        raise SystemExit("refusing --require-hardware run: manifest looks simulated")
    if not any(token in source for token in ("cw", "analog", "ro", "tdc")):
        raise SystemExit(f"refusing --require-hardware run: unexpected trace_source={source!r}")
    if manifest.get("hardware_run") is False:
        raise SystemExit("refusing --require-hardware run: hardware_run is false")
    if not (manifest.get("scope_serial") or manifest.get("target_serial")
            or manifest.get("bitstream_sha256")):
        raise SystemExit("refusing --require-hardware run: no hardware provenance fields")
    fc = manifest.get("functional_check")
    if isinstance(fc, dict) and fc.get("passed") is False:
        raise SystemExit("refusing --require-hardware run: functional_check failed")


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
    # threshold, do not truncate: a grey truth in [0,1] would otherwise floor to 0
    truth_bin = (np.asarray(truth01) >= 0.5).astype(np.uint8)
    tp = float(np.sum((rec_bin == 1) & (truth_bin == 1)))
    fp = float(np.sum((rec_bin == 1) & (truth_bin == 0)))
    fn = float(np.sum((rec_bin == 0) & (truth_bin == 1)))
    prec = tp / max(tp + fp, 1.0)
    rec = tp / max(tp + fn, 1.0)
    # Wei et al. pixel-level distance on the 0..255 scale
    pixel_dist = float(np.mean(np.abs(recovered01 * 255.0 - truth01 * 255.0)))
    mssim = float(np.mean([ssim_map(r, t) for r, t in zip(recovered01, truth01)]))
    corr = float(np.corrcoef(np.asarray(recovered01).ravel(),
                             np.asarray(truth01).ravel())[0, 1])
    # Chance baselines and the chance-relative rescalings. score_all.py routes the
    # template attack through image_metrics too, so these have to live here and not
    # only in leakage_first_reconstruct.metrics, or cross-dataset runs scored through
    # score_all silently lose the only fields that make them comparable.
    _fg = float(np.mean(truth_bin != 0))
    _f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    _f1_chance = 2.0 * _fg / (1.0 + _fg) if _fg > 0 else 0.0
    _acc = float(np.mean(rec_bin == truth_bin))
    _acc_chance = max(1.0 - _fg, _fg)
    return {
        "grey_mae_255": pixel_dist,
        "grey_rmse_255": float(np.sqrt(np.mean(
            (np.asarray(recovered01) * 255.0 - np.asarray(truth01) * 255.0) ** 2))),
        "grey_corr": corr,
        "bit_acc": float(np.mean(rec_bin == truth_bin)),
        "foreground_precision": prec,
        "foreground_recall": rec,
        "foreground_f1": 2 * prec * rec / max(prec + rec, 1e-9),
        "foreground_iou": tp / max(tp + fp + fn, 1.0),
        "all_zero_bit_acc": float(np.mean(truth_bin == 0)),
        "foreground_rate": _fg,
        # see leakage_first_reconstruct.metrics for why these exist
        "f1_chance": _f1_chance,
        "f1_excess": ((_f1 - _f1_chance) / (1.0 - _f1_chance)
                      if _f1_chance < 1 else 0.0),
        "bit_acc_chance": _acc_chance,
        "bit_acc_excess": ((_acc - _acc_chance) / (1.0 - _acc_chance)
                           if _acc_chance < 1 else 0.0),
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
    ap.add_argument("--split", choices=["random", "sequential"], default="random",
                    help="random split avoids acquisition-order/time drift confounding")
    ap.add_argument("--require-hardware", action="store_true",
                    help="fail if the capture manifest is not a live hardware capture")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--low-memory", action="store_true",
                    help="load rows in split order so the splits are views, not copies")
    ap.add_argument("--mmap-features", action="store_true",
                    help="back the feature array with a file instead of RAM")
    args = ap.parse_args()

    import torch
    import torch.nn as nn

    if args.threads:
        torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.out, exist_ok=True)

    # Featurize while loading: the raw array for a 60000-trace capture is 4.95 GiB
    # in float32, which will not fit alongside training.
    if args.low_memory:
        if args.split != "random" or args.limit:
            raise SystemExit("--low-memory requires --split random and no --limit")
        _mf = json.load(open(os.path.join(args.traces, "capture_manifest.json")))
        _n = int(_mf["n_images"])
        _order = np.random.default_rng(args.seed).permutation(_n)
        feats, images, labels, indices, manifest = load_shards_permuted(
            args.traces, args.features, _order)
    else:
        feats, images, labels, indices, manifest = load_shards(
            args.traces, args.limit, feature_mode=args.features,
            mmap_path=(os.path.join(args.out, "_feats.npy")
                       if args.mmap_features else None))
    if args.require_hardware:
        assert_hardware_manifest(manifest)
    # A grey capture stores the real 0..255 pixel, a binary one stores 0/1.  The
    # generator regresses into [0,1] either way, so scale the grey case down and
    # record which it was, because the metrics below need to know.
    grey_input = (manifest.get("input_mode") == "grey") or int(images.max()) > 1
    targets = images.astype(np.float32)
    if grey_input:
        targets /= 255.0
    n = len(feats)
    n_test = args.test_count
    n_val = args.val_count
    n_train = n - n_val - n_test
    if n_train <= 0:
        raise SystemExit("not enough captured images for the requested split")

    if args.low_memory:
        # rows are already stored in permutation order, so the splits are ranges
        tr = np.arange(0, n_train)
        va = np.arange(n_train, n_train + n_val)
        te = np.arange(n_train + n_val, n)
    elif args.split == "random":
        order = np.random.default_rng(args.seed).permutation(n)
        tr = order[:n_train]
        va = order[n_train:n_train + n_val]
        te = order[n_train + n_val:]
    else:
        tr = np.arange(0, n_train)
        va = np.arange(n_train, n_train + n_val)
        te = np.arange(n_train + n_val, n)

    # Chunked so nothing materialises a second full-size array. Mathematically the
    # same as feats[tr].mean(0) / .std(0) then (feats - mu) / sd; verified equal to
    # the direct computation before use.
    _tr = np.sort(tr)
    _s = np.zeros(feats.shape[1], dtype=np.float64)
    for c in range(0, len(_tr), 4096):
        _s += np.asarray(feats[_tr[c:c + 4096]], dtype=np.float64).sum(axis=0)
    _mu64 = _s / len(_tr)
    # Second pass about the mean rather than E[x^2]-E[x]^2: the one-pass form is
    # cheaper but loses precision, and these statistics have to match the other
    # grid cells, which were computed with numpy's own two-pass std.
    _v = np.zeros(feats.shape[1], dtype=np.float64)
    for c in range(0, len(_tr), 4096):
        d = np.asarray(feats[_tr[c:c + 4096]], dtype=np.float64) - _mu64
        _v += (d * d).sum(axis=0)
        del d
    mu = _mu64.astype(np.float32)
    sd = np.sqrt(_v / len(_tr)).astype(np.float32) + 1e-9
    for c in range(0, len(feats), 4096):
        feats[c:c + 4096] = (feats[c:c + 4096] - mu) / sd

    dev = torch.device("cpu")
    # Slices, not the index arrays: feats[np.arange(a, b)] is fancy indexing and
    # copies, which for the training split is another 1.13 GiB and defeats the
    # point of loading in split order. feats[a:b] is a view.
    if args.low_memory:
        _st = slice(0, n_train)
        _sv = slice(n_train, n_train + n_val)
        _se = slice(n_train + n_val, n)
    else:
        _st, _sv, _se = tr, va, te
    xt = torch.from_numpy(feats[_st]).float()
    yt = torch.from_numpy(targets[_st]).float()
    xv = torch.from_numpy(feats[_sv]).float()
    yv = torch.from_numpy(targets[_sv]).float()
    xe = torch.from_numpy(feats[_se]).float()

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
            total += float(loss.detach()) * len(idx)
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
        pred_orig = clf.predict((truth >= 0.5).astype(np.float32))
        pred_rec = clf.predict((rec >= 0.5).astype(np.float32))
        pred_rec_gray = clf.predict(np.clip(rec, 0, 1))
        m["recognition_accuracy_original"] = float(np.mean(pred_orig == lab))
        m["recognition_accuracy_recovered"] = float(np.mean(pred_rec == lab))
        m["recognition_accuracy_recovered_gray"] = float(np.mean(pred_rec_gray == lab))
    except Exception as exc:  # recognition is optional
        m["recognition_error"] = str(exc)

    np.savez_compressed(os.path.join(args.out, "test_recovered.npz"),
                        recovered=rec.astype(np.float32),
                        # float, not uint8: a grey truth lives in [0,1] and casting
                        # it to uint8 would floor every pixel to 0
                        truth=truth.astype(np.float32),
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
        "input_mode": "grey" if grey_input else "binary",
        "loss": args.loss,
        "split": args.split,
        "train_indices": indices[tr].astype(int).tolist(),
        "val_indices": indices[va].astype(int).tolist(),
        "test_indices": indices[te].astype(int).tolist(),
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
