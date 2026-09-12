#!/usr/bin/env python3
"""Power2Picture-style 3-channel generative input recovery for the CW305 RGB core.

Follows Huegle et al. (FCCM 2023) but differs in the two ways that matter for colour.

1. THREE OUTPUT CHANNELS, AND THAT PROVES NOTHING BY ITSELF.
   Their generator emits one channel because its last transposed convolution has one
   output feature map (their Table I). Setting that to three changes a tensor shape and
   nothing about what the trace contains. So this module refuses to report a headline
   number without the controls below -- producing a plausible RGB image is not evidence
   that colour was recovered.

2. CONTROLS ARE FIRST-CLASS, NOT OPTIONAL.
   Neither supplied paper reports a prior-only baseline. Without one, "the reconstruction
   looks like the input" cannot be separated from "the reconstruction looks like the
   dataset". Four arms are defined here and `run_all_arms` runs them together:

     trace            real traces, real pairing.
     prior_only       the optimal constant predictor, computed ANALYTICALLY as the
                      training-set mean image. With no per-sample input a model can only
                      emit a constant, and the MSE-optimal constant IS that mean, so
                      training a network to discover it can only do worse. This is the
                      number the trace arm must beat.

                      It is computed rather than trained because the trained version was
                      WRONG: feeding every row an identical feature vector makes the
                      feature standard deviation zero, so (x-mu)/(sd+1e-8) is all zeros
                      and BatchNorm sees zero variance. The network never converged and
                      scored MAE 122 where the true optimal constant scores 54 -- which
                      inflated the apparent advantage of the trace arm by ~68 MAE. Any
                      degenerate-input baseline has to be checked against its analytic
                      value, not assumed to train.
     shuffled         real traces paired with the WRONG images. Preserves every marginal
                      and destroys only the association. Should match prior_only; if it
                      does not, something other than the trace is leaking.
     channel_permuted scored at evaluation time against channel-swapped truth. If the
                      error does not rise, the model is not recovering channel identity.

THE METRIC THAT MATTERS FOR COLOUR
----------------------------------
Natural-image colour is strongly predictable from luminance and semantics. A model that
recovers structure from the side channel and then colourises it from the training prior
would score well on pooled PSNR and look convincing. So errors are reported per channel
and split into luminance (Y) and chrominance (Cb, Cr). Recovering Y while failing Cb/Cr
is "grayscale recovery from an RGB-input accelerator" -- a real and publishable finding,
and explicitly NOT RGB reconstruction. Pooled RGB PSNR hides exactly this, so it is
reported alongside but never alone.

Data comes from host/cw305_rgb_capture.py, which writes one subdirectory per dataflow
mode. Modes are compared only against each other at equal trace length.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import time

import numpy as np

RESULT_KIND = "cw305_rgb_power2picture_generator"
ARMS = ("trace", "prior_only", "shuffled")


# ---------------------------------------------------------------- data


def load_rgb_shards(trace_dir, mode, limit=None, feature="settle"):
    """Load one dataflow mode's capture. Returns (feats, images, indices, manifest)."""
    manifest = json.load(open(os.path.join(trace_dir, "capture_manifest.json")))
    if mode not in manifest.get("modes", {}):
        raise SystemExit(f"mode {mode!r} not in capture; have {list(manifest.get('modes', {}))}")
    mdir = os.path.join(trace_dir, manifest["modes"][mode]["dir"])
    paths = sorted(glob.glob(os.path.join(mdir, "shard*.npz")))
    if not paths:
        raise SystemExit(f"no shard*.npz in {mdir}")

    fc = manifest["modes"][mode].get("functional_check", {})
    if not fc.get("passed", False):
        raise SystemExit(
            f"mode {mode!r} functional check did not pass ({fc.get('mismatches')} "
            "mismatches). Refusing to train on traces whose design did not verify.")

    scale = float(manifest.get("trace_scale", 32767.0))
    feats, imgs, idxs = [], [], []
    total = 0
    for p in paths:
        d = np.load(p)
        t = d["traces"].astype(np.float32) / scale
        f = featurize_rgb(t, manifest, feature)
        take = len(f) if limit is None else max(0, min(len(f), limit - total))
        if take <= 0:
            break
        feats.append(f[:take]); imgs.append(d["images"][:take]); idxs.append(d["indices"][:take])
        total += take
        del t, d
    return (np.concatenate(feats), np.concatenate(imgs),
            np.concatenate(idxs), manifest)


def featurize_rgb(traces, manifest, mode="settle"):
    """Reduce raw ADC samples to a generator input vector.

    `settle` is the default because it is measured to be better: on the reference grey
    capture the data-dependent activity is confined to a few samples early in each dwell
    period (corr 0.306 at sample 8, ~0.00 after sample 16), so averaging the whole dwell
    dilutes it. Samples 6..13 scored 0.347 against 0.236 for the full-window mean -- a
    1.47x correlation gain for free. See experiments/rgb_sca/rgb_bench.py.

    The settle window is expressed in samples-per-window units so it tracks dwell and
    samples_per_cycle rather than being hardcoded.
    """
    spc = int(manifest["samples_per_cycle"])
    dwell = int(manifest["dwell"])
    spw = dwell * spc
    x = traces - np.median(traces, axis=1, keepdims=True)
    n = x.shape[1] // spw
    x = np.abs(x[:, :n * spw]).reshape(len(x), n, spw)
    if mode == "window_abs":
        return x.mean(axis=2)
    if mode == "settle":
        lo = max(0, (6 * spw) // 32)
        hi = max(lo + 1, (14 * spw) // 32)
        return x[:, :, lo:hi].mean(axis=2)
    if mode == "full":
        return x.reshape(len(x), n * spw)
    raise ValueError(f"unknown feature mode {mode}")


# ---------------------------------------------------------------- metrics


def _ssim_box11(a, b, data_range=1.0, win=11):
    """Mean SSIM over UNIFORM (box) 11x11 windows -- named for what it is.

    Wang et al.'s reference implementation uses GAUSSIAN weighting, so this is a defined
    SSIM variant, not the canonical metric, and values are not directly comparable to
    library scores. Renamed from `_ssim_map` after external review flagged the mismatch
    between the name and the description. All reported MSSIM figures in this project use
    this box variant; the relative comparisons between arms are unaffected because every
    arm is scored identically.

    The window is clamped to the image when the image is smaller than 11 px (small
    synthetic cases in the tests), since sliding_window_view raises rather than degrading
    gracefully. Clamping keeps the metric defined; it is never widened past 11.
    """
    from numpy.lib.stride_tricks import sliding_window_view
    win = min(win, a.shape[-2], a.shape[-1])
    if win < 2:
        return float("nan")
    c1, c2 = (0.01 * data_range) ** 2, (0.03 * data_range) ** 2
    aw = sliding_window_view(a, (win, win)).reshape(-1, win * win)
    bw = sliding_window_view(b, (win, win)).reshape(-1, win * win)
    ma, mb = aw.mean(1), bw.mean(1)
    va, vb = aw.var(1), bw.var(1)
    cov = (aw * bw).mean(1) - ma * mb
    return float(np.mean(((2 * ma * mb + c1) * (2 * cov + c2)) /
                         ((ma ** 2 + mb ** 2 + c1) * (va + vb + c2))))


def _to_ycbcr(x01):
    """(N,3,H,W) in [0,1] -> Y, Cb, Cr each (N,H,W). ITU-R BT.601."""
    r, g, b = x01[:, 0], x01[:, 1], x01[:, 2]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = 0.564 * (b - y)
    cr = 0.713 * (r - y)
    return y, cb, cr


def rgb_metrics(rec01, truth01):
    """Per-channel and luminance/chrominance errors. Never pools the channels away."""
    rec01 = np.clip(np.asarray(rec01, dtype=np.float64), 0.0, 1.0)
    truth01 = np.asarray(truth01, dtype=np.float64)
    d = (rec01 - truth01) * 255.0

    per_ch_mae = [float(np.mean(np.abs(d[:, c]))) for c in range(3)]
    per_ch_mse = [float(np.mean(d[:, c] ** 2)) for c in range(3)]
    per_ch_psnr = [float(10 * np.log10(255.0 ** 2 / m)) if m > 0 else float("inf")
                   for m in per_ch_mse]

    ry, rcb, rcr = _to_ycbcr(rec01)
    ty, tcb, tcr = _to_ycbcr(truth01)
    # Chrominance is scaled to the same 0..255 footing as Y so the two are comparable.
    y_mae = float(np.mean(np.abs(ry - ty)) * 255.0)
    cb_mae = float(np.mean(np.abs(rcb - tcb)) * 255.0)
    cr_mae = float(np.mean(np.abs(rcr - tcr)) * 255.0)

    pooled_mse = float(np.mean(d ** 2))
    return {
        "mae_per_channel": per_ch_mae,
        "psnr_per_channel": per_ch_psnr,
        "mae_pooled": float(np.mean(np.abs(d))),
        "psnr_pooled": (float(10 * np.log10(255.0 ** 2 / pooled_mse))
                        if pooled_mse > 0 else float("inf")),
        # The split that separates "recovered the picture" from "colourised the picture"
        "luma_mae": y_mae,
        "chroma_mae": [cb_mae, cr_mae],
        "chroma_over_luma": float((cb_mae + cr_mae) / 2.0 / max(y_mae, 1e-9)),
        "mssim_per_channel": [
            float(np.mean([_ssim_box11(r, t) for r, t in zip(rec01[:, c], truth01[:, c])]))
            for c in range(3)],
        "corr_per_channel": [
            float(np.corrcoef(rec01[:, c].ravel(), truth01[:, c].ravel())[0, 1])
            for c in range(3)],
    }


def channel_permutation_check(rec01, truth01, perm=(2, 1, 0), prior01=None):
    """Score against channel-swapped truth.

    If the error does not RISE relative to the correct pairing, the reconstruction is not
    carrying channel identity -- it is producing a colour field that fits either
    assignment about equally well. Reported as a ratio so it is scale-free.
    """
    correct = rgb_metrics(rec01, truth01)["mae_pooled"]
    swapped = rgb_metrics(rec01, np.asarray(truth01)[:, list(perm)])["mae_pooled"]
    out = {
        "mae_correct_pairing": correct,
        "mae_swapped_pairing": swapped,
        "swap_penalty": float(swapped - correct),
        "swap_penalty_ratio": float(swapped / max(correct, 1e-9)),
        "interpretation": ("swap_penalty near zero means channel identity was NOT "
                           "recovered, regardless of how good the pooled score looks"),
    }
    # A POSITIVE swap penalty is not by itself evidence of measurement-derived colour.
    # An input-independent constant predictor incurs one too whenever the dataset's
    # channel distributions differ: on CIFAR the mean image alone scores +1.04, which is
    # two thirds of the +1.58 the natural-group reconstruction scores. So the informative
    # quantity is the EXCESS over the no-input prior. Raised by external review,
    # 12 September 2026.
    if prior01 is not None:
        pc = rgb_metrics(prior01, truth01)["mae_pooled"]
        ps = rgb_metrics(prior01, np.asarray(truth01)[:, list(perm)])["mae_pooled"]
        out["prior_swap_penalty"] = float(ps - pc)
        out["prior_swap_penalty_ratio"] = float(ps / max(pc, 1e-9))
        out["excess_swap_penalty_over_prior"] = float((swapped - correct) - (ps - pc))
        out["interpretation"] += ("; compare against prior_swap_penalty -- only "
                                  "excess_swap_penalty_over_prior is attributable to "
                                  "the measurement")
    return out


# ---------------------------------------------------------------- model


def build_rgb_learned_filter(n_windows, spw, out_side, torch, nn):
    """Generator with a LEARNED per-window temporal filter instead of a fixed window.

    The hand-picked featurization averages ADC samples 6..13 of each dwell period. That
    choice dominates everything measured: correlation ranges from 0.011 (one sample) to
    0.333 (samples 7..10) to 0.188 (all 32) on the same traces. Picking that window by
    hand is leaving information on the table, and the optimum is unlikely to be a flat
    average over a hard boundary.

    So the first stage is a depthwise Conv1d over the sample axis that learns its own
    weighting of the samples-per-window, reducing (n_windows, spw) -> (n_windows). The
    rest of the network is unchanged, so any gain is attributable to the filter alone.

    This costs nothing in hardware and needs no recapture -- the raw samples are already
    in the traces we have.
    """
    if out_side % 4 != 0:
        raise ValueError("out_side must be divisible by 4 for the 2-stage upsample")
    seed = out_side // 4
    flat = 256 * seed * seed

    class LearnedFilterGenerator(nn.Module):
        def __init__(self):
            super().__init__()
            self.spw = spw
            self.n_windows = n_windows
            # One shared temporal filter across windows: the settling transient has the
            # same shape in every dwell period, so sharing is the right inductive bias
            # and keeps the parameter count negligible.
            self.tfilt = nn.Conv1d(1, 4, kernel_size=spw, stride=spw, bias=True)
            self.mix = nn.Linear(4 * n_windows, n_windows)
            self.seed = seed
            self.fc = nn.Sequential(
                nn.Linear(n_windows, 128), nn.BatchNorm1d(128), nn.LeakyReLU(0.2),
                nn.Linear(128, 128), nn.BatchNorm1d(128), nn.LeakyReLU(0.2),
                nn.Linear(128, flat), nn.BatchNorm1d(flat), nn.LeakyReLU(0.2),
            )
            self.deconv = nn.Sequential(
                nn.ConvTranspose2d(256, 128, 5, stride=1, padding=2),
                nn.BatchNorm2d(128), nn.LeakyReLU(0.2),
                nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
                nn.BatchNorm2d(64), nn.LeakyReLU(0.2),
                nn.ConvTranspose2d(64, 3, 4, stride=2, padding=1),
            )

        def forward(self, x):                       # x: (N, n_windows * spw)
            b = x.shape[0]
            h = self.tfilt(x.view(b, 1, -1))        # (N, 4, n_windows)
            h = self.mix(h.reshape(b, -1))          # (N, n_windows)
            h = self.fc(h).view(-1, 256, self.seed, self.seed)
            return torch.sigmoid(self.deconv(h))

    return LearnedFilterGenerator()


def build_rgb_model(in_dim, out_side, torch, nn):
    """Power2Picture generator widened to 3 channels and the RGB output size.

    Their Table I seeds a 7x7 feature map and upsamples 7 -> 14 -> 28. For a 32x32 output
    the seed is 8x8 and the path is 8 -> 16 -> 32; the layer types, kernel sizes, strides
    and padding are unchanged, so this stays a faithful widening rather than a redesign.
    """
    if out_side % 4 != 0:
        raise ValueError("out_side must be divisible by 4 for the 2-stage upsample")
    seed = out_side // 4
    flat = 256 * seed * seed

    class Generator(nn.Module):
        def __init__(self):
            super().__init__()
            self.seed = seed
            self.fc = nn.Sequential(
                nn.Linear(in_dim, 128), nn.BatchNorm1d(128), nn.LeakyReLU(0.2),
                nn.Linear(128, 128), nn.BatchNorm1d(128), nn.LeakyReLU(0.2),
                nn.Linear(128, flat), nn.BatchNorm1d(flat), nn.LeakyReLU(0.2),
            )
            self.deconv = nn.Sequential(
                nn.ConvTranspose2d(256, 128, 5, stride=1, padding=2),
                nn.BatchNorm2d(128), nn.LeakyReLU(0.2),
                nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
                nn.BatchNorm2d(64), nn.LeakyReLU(0.2),
                nn.ConvTranspose2d(64, 3, 4, stride=2, padding=1),   # 3 channels
            )

        def forward(self, x):
            x = self.fc(x).view(-1, 256, self.seed, self.seed)
            return torch.sigmoid(self.deconv(x))                     # (N,3,H,W)

    return Generator()


def gradmse(pred, target, torch, mse):
    """Their Eq. 1, applied per channel: MSE plus the x and y image-gradient MSEs."""
    loss = mse(pred, target)
    loss = loss + mse(pred[:, :, :, 1:] - pred[:, :, :, :-1],
                      target[:, :, :, 1:] - target[:, :, :, :-1])
    loss = loss + mse(pred[:, :, 1:, :] - pred[:, :, :-1, :],
                      target[:, :, 1:, :] - target[:, :, :-1, :])
    return loss


# ---------------------------------------------------------------- image export


def save_reconstructions(recs, out_dir, mode, n_show=12, seed=0):
    """Save the recovered RGB images -- the actual deliverable, not just the metrics.

    The headline result of both attacked papers is an image grid (Wei et al. Fig. 11,
    Huegle et al. Fig. 4), so this writes one. It deliberately puts THREE rows side by
    side rather than the usual two:

        row 1  original input
        row 2  recovered from the trace
        row 3  recovered by the PRIOR-ONLY control (no trace information at all)

    Row 3 is the one that makes the figure honest. A two-row figure invites the reader to
    credit the side channel for whatever the reconstruction got right, when a model
    trained on the same dataset may produce something similar with no measurement at all.
    Printing the control underneath makes that comparison unavoidable, and it is the
    first thing a sceptical reviewer would ask for.

    Also writes the raw float arrays so figures can be regenerated without retraining.
    """
    from PIL import Image

    os.makedirs(out_dir, exist_ok=True)
    rec_t, truth, te = recs["trace"]
    rec_p = recs["prior_only"][0] if "prior_only" in recs else None

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(truth), size=min(n_show, len(truth)), replace=False)

    def _u8(a):
        return (np.clip(a, 0, 1) * 255).round().astype(np.uint8).transpose(1, 2, 0)

    rows = [truth[idx], rec_t[idx]] + ([rec_p[idx]] if rec_p is not None else [])
    h, w = truth.shape[-2], truth.shape[-1]
    pad = 2
    grid = np.full(((h + pad) * len(rows) + pad, (w + pad) * len(idx) + pad, 3),
                   255, dtype=np.uint8)
    for r, row in enumerate(rows):
        for c in range(len(idx)):
            y0 = pad + r * (h + pad)
            x0 = pad + c * (w + pad)
            grid[y0:y0 + h, x0:x0 + w] = _u8(row[c])

    scale = max(1, 256 // h)
    img = Image.fromarray(grid)
    img = img.resize((grid.shape[1] * scale, grid.shape[0] * scale), Image.NEAREST)
    grid_path = os.path.join(out_dir, f"comparison_{mode}.png")
    img.save(grid_path)

    np.savez_compressed(
        os.path.join(out_dir, f"reconstructions_{mode}.npz"),
        truth=truth.astype(np.float32), trace=rec_t.astype(np.float32),
        prior_only=(rec_p.astype(np.float32) if rec_p is not None else np.zeros(0)),
        test_indices=np.asarray(te), shown=idx)

    rows_desc = "original / recovered-from-trace" + ("/ prior-only control"
                                                    if rec_p is not None else "")
    return {"grid": grid_path, "rows": rows_desc, "n_shown": int(len(idx))}


# ---------------------------------------------------------------- arms


def make_arm_features(feats, arm, rng, tr=None, te=None):
    """Transform the input features for one control arm. Targets are never touched.

    `prior_only` is NOT handled here -- it is computed analytically in train_arm, because
    a degenerate all-identical input cannot be standardized or BatchNormed. See the module
    docstring.

    SHUFFLING RESPECTS THE SPLIT. An earlier version permuted rows across the WHOLE
    dataset, so a test row could receive a training row's features -- in a 2,000-row
    reproduction, 185 held-out rows moved to training positions. The control's job is to
    break the trace/image association, not to move data across the partition, so each
    partition is now permuted independently. Found in external review, 12 Sep 2026.
    """
    if arm == "trace":
        return feats
    if arm == "shuffled":
        out = feats.copy()
        if tr is None or te is None:
            return feats[rng.permutation(len(feats))]
        for part in (np.asarray(tr), np.asarray(te)):
            if len(part) > 1:
                out[part] = feats[part][rng.permutation(len(part))]
        return out
    if arm == "prior_only":
        raise ValueError("prior_only is computed analytically; see analytic_prior()")
    raise ValueError(f"unknown arm {arm}")


def analytic_prior(y_train, n_test):
    """The MSE-optimal constant predictor: the training-set mean image, broadcast.

    Exact, instant, and cannot fail to converge -- the three properties a baseline needs.
    """
    return np.repeat(y_train.mean(axis=0, keepdims=True), n_test, axis=0)


def train_arm(feats, images, arm, args, torch, nn, rng):
    """Train one arm and score it on a held-out split. Split is by image index."""
    n = len(feats)
    order = rng.permutation(n)
    n_te = max(1, int(round(args.test_frac * n)))
    te, tr = order[:n_te], order[n_te:]

    y = (images.astype(np.float32) / 255.0)

    if arm == "prior_only":
        # No training: the optimal constant predictor is closed-form. Returning early
        # also sidesteps the zero-variance standardization that silently broke this arm.
        rec = analytic_prior(y[tr], len(te))
        truth = y[te]
        return ({"arm": arm, "n_train": int(len(tr)), "n_test": int(len(te)),
                 "method": "analytic_training_mean", **rgb_metrics(rec, truth)},
                rec, truth, te)

    x = make_arm_features(feats, arm, rng, tr=tr, te=te).astype(np.float32)
    mu, sd = x[tr].mean(0), x[tr].std(0) + 1e-8
    x = (x - mu) / sd

    # Seed torch too. The requested seed previously reached only the NumPy split, so
    # model initialization and batch permutation were unseeded and a rerun at "seed=0"
    # was not reproducible -- which made small cross-mode differences uninterpretable.
    # Full determinism still depends on platform, threading and library version; see
    # https://docs.pytorch.org/docs/stable/notes/randomness.html
    torch.manual_seed(args.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_rgb_model(x.shape[1], images.shape[-1], torch, nn).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    mse = nn.MSELoss()

    xt = torch.from_numpy(x[tr]).to(dev)
    yt = torch.from_numpy(y[tr]).to(dev)
    model.train()
    for ep in range(args.epochs):
        perm = torch.randperm(len(xt), device=dev)
        tot = 0.0
        for i in range(0, len(xt), args.batch):
            idx = perm[i:i + args.batch]
            if len(idx) < 2:            # BatchNorm needs >1 sample
                continue
            opt.zero_grad()
            loss = (gradmse(model(xt[idx]), yt[idx], torch, mse) if args.loss == "gradmse"
                    else mse(model(xt[idx]), yt[idx]))
            loss.backward(); opt.step()
            tot += float(loss.detach()) * len(idx)
        if (ep + 1) % max(1, args.epochs // 5) == 0:
            print(f"    {arm} epoch {ep+1}/{args.epochs} loss {tot/max(len(xt),1):.5f}")

    model.eval()
    with torch.no_grad():
        rec = []
        xe = torch.from_numpy(x[te]).to(dev)
        for i in range(0, len(xe), args.batch):
            rec.append(model(xe[i:i + args.batch]).cpu().numpy())
    rec = np.concatenate(rec)
    truth = y[te]
    out = {"arm": arm, "n_train": int(len(tr)), "n_test": int(len(te)),
           **rgb_metrics(rec, truth)}
    if arm == "trace":
        out["channel_permutation"] = channel_permutation_check(
            rec, truth, prior01=analytic_prior(y[tr], len(te)))
    return out, rec, truth, te


def run_all_arms(feats, images, args, torch, nn):
    """Run every arm on the same split seed so the comparison is paired."""
    results = {}
    recs = {}
    for arm in ARMS:
        print(f"  arm: {arm}")
        rng = np.random.default_rng(args.seed)      # same split for every arm
        res, rec, truth, te = train_arm(feats, images, arm, args, torch, nn, rng)
        results[arm] = res
        recs[arm] = (rec, truth, te)

    t, p = results["trace"], results["prior_only"]
    results["summary"] = {
        "advantage_over_prior_pooled_mae": float(p["mae_pooled"] - t["mae_pooled"]),
        "advantage_over_prior_per_channel": [
            float(a - b) for a, b in zip(p["mae_per_channel"], t["mae_per_channel"])],
        "luma_advantage": float(p["luma_mae"] - t["luma_mae"]),
        "chroma_advantage": [float(a - b) for a, b in
                             zip(p["chroma_mae"], t["chroma_mae"])],
        "shuffled_matches_prior": bool(
            abs(results["shuffled"]["mae_pooled"] - p["mae_pooled"])
            < 0.05 * max(p["mae_pooled"], 1e-9)),
        "reading": ("A positive pooled advantage with near-zero chroma advantage means "
                    "luminance was recovered and colour came from the prior. Report that "
                    "as grayscale recovery from an RGB-input accelerator, not as RGB "
                    "reconstruction."),
    }
    return results, recs


# ---------------------------------------------------------------- main


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trace-dir", required=True, help="output of cw305_rgb_capture.py")
    ap.add_argument("--modes", nargs="*", default=["serial", "parallel", "summed"])
    ap.add_argument("--feature", default="settle",
                    choices=["settle", "window_abs", "full"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--loss", choices=["gradmse", "mse"], default="gradmse")
    ap.add_argument("--test-frac", type=float, default=0.1)
    ap.add_argument("--n-show", type=int, default=12,
                    help="images per row in the comparison grid")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import torch
    from torch import nn

    t0 = time.time()
    all_results = {
        "kind": RESULT_KIND,
        "source": "hardware",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "args": vars(args),
        "modes": {},
    }

    for mode in args.modes:
        print(f"\nmode {mode}")
        feats, images, _idx, manifest = load_rgb_shards(
            args.trace_dir, mode, args.limit, args.feature)
        print(f"  features {feats.shape}  images {images.shape}")
        all_results.setdefault("capture", {
            "bitstream_sha256": manifest.get("bitstream_sha256"),
            "design_signature": manifest.get("design_signature"),
            "capture_mode": manifest.get("capture_mode"),
            "avg": manifest.get("avg"),
            "dwell": manifest.get("dwell"),
            "samples_per_cycle": manifest.get("samples_per_cycle"),
        })
        res, recs = run_all_arms(feats, images, args, torch, nn)
        all_results["modes"][mode] = res
        if args.out:
            saved = save_reconstructions(recs, os.path.join(args.out, "images"), mode,
                                         n_show=args.n_show, seed=args.seed)
            res["images"] = saved
            print(f"    images: {saved['grid']}  ({saved['rows']})")

        s, t, p = res["summary"], res["trace"], res["prior_only"]
        print(f"    trace  mae/ch {[round(v,2) for v in t['mae_per_channel']]}"
              f"  luma {t['luma_mae']:.2f}  chroma {[round(v,2) for v in t['chroma_mae']]}")
        print(f"    prior  mae/ch {[round(v,2) for v in p['mae_per_channel']]}"
              f"  luma {p['luma_mae']:.2f}  chroma {[round(v,2) for v in p['chroma_mae']]}")
        print(f"    advantage pooled {s['advantage_over_prior_pooled_mae']:+.2f}"
              f"  luma {s['luma_advantage']:+.2f}"
              f"  chroma {[round(v,2) for v in s['chroma_advantage']]}")
        print(f"    shuffled matches prior: {s['shuffled_matches_prior']}")
        cp = t["channel_permutation"]
        print(f"    channel-swap penalty {cp['swap_penalty']:+.2f}"
              f" (ratio {cp['swap_penalty_ratio']:.3f})")

    all_results["elapsed_sec"] = round(time.time() - t0, 2)
    if args.out:
        os.makedirs(args.out, exist_ok=True)
        with open(os.path.join(args.out, "results.json"), "w") as fh:
            json.dump(all_results, fh, indent=1, default=float)
        print(f"\nwrote {os.path.join(args.out, 'results.json')}")


if __name__ == "__main__":
    main()
