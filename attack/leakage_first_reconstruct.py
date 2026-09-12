#!/usr/bin/env python3
"""Reconstruct 28x28 binary MNIST inputs from cw305_leakage_top traces.

Profile known traces learn feature-vs-match-count lookup tables per kernel.
Evaluation traces are decoded window by window over all 512 possible 3x3 binary
patches, then overlapping recovered patches majority-vote the full image.
"""
import argparse
import glob
import json
import os

import numpy as np

LINE = 28
OUT_SIDE = 26
N_WINDOWS = OUT_SIDE * OUT_SIDE
N_PATCH_CODES = 512


def load_kernel_bits(path, n_kernels, probe_kernels="file"):
    if probe_kernels == "onehot":
        if n_kernels != 9:
            raise ValueError("onehot probe mode requires 9 kernels")
        return np.eye(9, dtype=np.uint8)
    kernels = np.load(path)
    if kernels.ndim != 3 or kernels.shape[1:] != (3, 3):
        raise ValueError(f"expected kernels shaped (N,3,3), got {kernels.shape}")
    return (kernels[:n_kernels].reshape(n_kernels, 9) > 0).astype(np.uint8)


def patch_matrix_from_image(img_bits):
    patches = np.zeros((N_WINDOWS, 9), dtype=np.uint8)
    idx = 0
    for y in range(OUT_SIDE):
        for x in range(OUT_SIDE):
            patches[idx] = img_bits[y:y + 3, x:x + 3].reshape(9)
            idx += 1
    return patches


def patch_codes_from_patches(patches):
    weights = (1 << np.arange(9, dtype=np.uint16))
    return (patches.astype(np.uint16) * weights).sum(axis=1).astype(np.uint16)


def all_patch_bits():
    codes = np.arange(N_PATCH_CODES, dtype=np.uint16)
    return (((codes[:, None] >> np.arange(9, dtype=np.uint16)) & 1)
            .astype(np.uint8))


def match_counts_for_codes(kernel_bits):
    bits = all_patch_bits()
    return np.stack([
        np.count_nonzero(bits == kernel_bits[k][None, :], axis=1)
        for k in range(kernel_bits.shape[0])
    ], axis=0).astype(np.int16)


def robust_norm(feat):
    med = np.median(feat, axis=1, keepdims=True)
    mad = np.median(np.abs(feat - med), axis=1, keepdims=True)
    scale = np.maximum(1.4826 * mad, 1e-6)
    return (feat - med) / scale


def extract_features(npz_path, dwell, spc, presamples, mode):
    """Per-window power features, shape (rows, N_WINDOWS).

    Normally one row per probe kernel.  Mode ``cycle_abs`` keeps one row per
    (kernel, FPGA clock cycle) instead, which is closer to the per-clock-cycle
    power extraction of Wei et al. than a single per-window scalar.
    """
    if mode == "cycle_abs":
        return extract_cycle_features(npz_path, dwell, spc, presamples)
    d = np.load(npz_path)
    traces = d["repeats"] if "repeats" in d.files else d["traces"][:, None, :]
    n_kernels = traces.shape[0]
    win_samp = dwell * spc
    feat = np.zeros((n_kernels, N_WINDOWS), dtype=np.float32)
    for kid in range(n_kernels):
        reps = traces[kid].astype(np.float32)
        reps = reps - np.median(reps, axis=1, keepdims=True)
        per_rep = np.zeros((reps.shape[0], N_WINDOWS), dtype=np.float32)
        for w in range(N_WINDOWS):
            a = presamples + w * win_samp
            b = a + win_samp
            seg = reps[:, a:b]
            if mode == "mean_abs":
                per_rep[:, w] = np.mean(np.abs(seg), axis=1)
            elif mode == "rms":
                per_rep[:, w] = np.sqrt(np.mean(seg * seg, axis=1))
            elif mode == "p2p":
                per_rep[:, w] = np.max(seg, axis=1) - np.min(seg, axis=1)
            else:
                raise ValueError(f"unknown feature mode {mode}")
        feat[kid] = np.median(per_rep, axis=0)
    return robust_norm(feat)


def extract_cycle_features(npz_path, dwell, spc, presamples):
    """One feature per (kernel, clock cycle) inside each convolution window."""
    d = np.load(npz_path)
    traces = d["repeats"] if "repeats" in d.files else d["traces"][:, None, :]
    n_kernels = traces.shape[0]
    win_samp = dwell * spc
    rows = np.zeros((n_kernels * dwell, N_WINDOWS), dtype=np.float32)
    for kid in range(n_kernels):
        reps = traces[kid].astype(np.float32)
        reps = reps - np.median(reps, axis=1, keepdims=True)
        usable = presamples + N_WINDOWS * win_samp
        seg = np.abs(reps[:, presamples:usable])
        seg = seg.reshape(reps.shape[0], N_WINDOWS, dwell, spc).mean(axis=3)
        per_cycle = np.median(seg, axis=0)              # (N_WINDOWS, dwell)
        rows[kid * dwell:(kid + 1) * dwell] = per_cycle.T
    return robust_norm(rows)


def fit_lookup(profile_files, kernel_bits, dwell, spc, presamples, feature_mode):
    n_kernels = kernel_bits.shape[0]
    buckets = [[[] for _ in range(10)] for _ in range(n_kernels)]
    for path in profile_files:
        d = np.load(path)
        feat = extract_features(path, dwell, spc, presamples, feature_mode)
        patches = patch_matrix_from_image(d["image"].astype(np.uint8))
        for kid in range(n_kernels):
            counts = np.count_nonzero(patches == kernel_bits[kid][None, :], axis=1)
            for c in range(10):
                vals = feat[kid, counts == c]
                if len(vals):
                    buckets[kid][c].extend(vals.astype(float).tolist())

    lookup = np.zeros((n_kernels, 10), dtype=np.float32)
    for kid in range(n_kernels):
        xs = []
        ys = []
        for c in range(10):
            if buckets[kid][c]:
                xs.append(c)
                ys.append(float(np.median(buckets[kid][c])))
        if len(xs) >= 2:
            lookup[kid] = np.interp(np.arange(10), xs, ys).astype(np.float32)
        elif len(xs) == 1:
            lookup[kid].fill(ys[0])
        else:
            raise RuntimeError(f"no profile samples for kernel {kid}")
    return lookup


def decode_patches(feat, lookup, code_match_counts):
    n_kernels = feat.shape[0]
    templates = np.zeros((N_PATCH_CODES, n_kernels), dtype=np.float32)
    for kid in range(n_kernels):
        templates[:, kid] = lookup[kid, code_match_counts[kid]]

    pred_codes = np.zeros(N_WINDOWS, dtype=np.uint16)
    for w in range(N_WINDOWS):
        diff = templates - feat[:, w][None, :]
        pred_codes[w] = int(np.argmin(np.mean(diff * diff, axis=1)))
    return pred_codes


def reconstruct_from_codes(codes):
    bits = all_patch_bits()
    votes = np.zeros((LINE, LINE), dtype=np.int16)
    counts = np.zeros((LINE, LINE), dtype=np.int16)
    idx = 0
    for y in range(OUT_SIDE):
        for x in range(OUT_SIDE):
            patch = bits[int(codes[idx])].reshape(3, 3)
            votes[y:y + 3, x:x + 3] += patch
            counts[y:y + 3, x:x + 3] += 1
            idx += 1
    return (votes * 2 >= counts).astype(np.uint8)


def metrics(pred, truth):
    pred = pred.astype(bool)
    truth = truth.astype(bool)
    tp = int(np.count_nonzero(pred & truth))
    tn = int(np.count_nonzero(~pred & ~truth))
    fp = int(np.count_nonzero(pred & ~truth))
    fn = int(np.count_nonzero(~pred & truth))
    bit_acc = (tp + tn) / pred.size
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-12)
    iou = tp / max(tp + fp + fn, 1)
    zero_acc = int(np.count_nonzero(~truth)) / truth.size

    # Chance-relative versions of the two headline numbers.
    #
    # Raw F1 and raw pixel accuracy are only comparable between runs that share
    # a foreground rate. MNIST is 13.4 % foreground and Fashion-MNIST is 31.5 %,
    # which moves the all-background pixel accuracy from 0.866 to 0.685 and the
    # best F1 reachable by a predictor that knows nothing from 0.237 to 0.479.
    # Comparing raw F1 across those two datasets therefore compares the datasets
    # and not the attack: an F1 of 0.55 is a good result on MNIST and barely
    # above chance on Fashion-MNIST. These fields rescale each metric so that
    # 0 is chance and 1 is perfect, and they are the ones to quote whenever the
    # foreground rate is not held fixed.
    fg_rate = 1.0 - zero_acc
    # A predictor marking every pixel foreground gets precision = fg_rate and
    # recall = 1, which maximises F1 over all label-blind predictors.
    f1_chance = 2.0 * fg_rate / (1.0 + fg_rate) if fg_rate > 0 else 0.0
    # Pixel accuracy's chance level is the better of always-background and
    # always-foreground.
    acc_chance = max(zero_acc, fg_rate)
    return {
        "bit_acc": bit_acc,
        "foreground_precision": prec,
        "foreground_recall": rec,
        "foreground_f1": f1,
        "foreground_iou": iou,
        "all_zero_bit_acc": zero_acc,
        "foreground_rate": fg_rate,
        "f1_chance": f1_chance,
        "f1_excess": (f1 - f1_chance) / (1.0 - f1_chance) if f1_chance < 1 else 0.0,
        "bit_acc_chance": acc_chance,
        "bit_acc_excess": ((bit_acc - acc_chance) / (1.0 - acc_chance)
                           if acc_chance < 1 else 0.0),
    }


def fit_lookup_from_features(feature_rows, kernel_bits):
    n_kernels = kernel_bits.shape[0]
    buckets = [[[] for _ in range(10)] for _ in range(n_kernels)]
    for feat, img in feature_rows:
        patches = patch_matrix_from_image(img)
        for kid in range(n_kernels):
            counts = np.count_nonzero(patches == kernel_bits[kid][None, :], axis=1)
            for c in range(10):
                vals = feat[kid, counts == c]
                if len(vals):
                    buckets[kid][c].extend(vals.astype(float).tolist())

    lookup = np.zeros((n_kernels, 10), dtype=np.float32)
    for kid in range(n_kernels):
        xs = []
        ys = []
        for c in range(10):
            if buckets[kid][c]:
                xs.append(c)
                ys.append(float(np.median(buckets[kid][c])))
        if len(xs) >= 2:
            lookup[kid] = np.interp(np.arange(10), xs, ys).astype(np.float32)
        elif len(xs) == 1:
            lookup[kid].fill(ys[0])
        else:
            raise RuntimeError(f"no profile samples for kernel {kid}")
    return lookup


def synthetic_selftest(args):
    d = np.load(args.images)
    imgs = d["images"]
    if imgs.ndim == 4 and imgs.shape[-1] == 1:
        imgs = imgs[..., 0]
    labels = d["labels"]
    kernel_bits = load_kernel_bits(args.kernels, args.n_kernels, args.probe_kernels)
    code_counts = match_counts_for_codes(kernel_bits)
    rng = np.random.default_rng(args.seed)
    outdir = args.out
    os.makedirs(outdir, exist_ok=True)
    raw_lookup = np.stack([np.linspace(-2.0, 2.0, 10) for _ in range(args.n_kernels)])
    feature_rows = []
    images = []
    codes_all = []
    for i in range(args.n_images):
        img = (imgs[i] > args.threshold).astype(np.uint8)
        patches = patch_matrix_from_image(img)
        codes = patch_codes_from_patches(patches)
        feat = np.zeros((args.n_kernels, N_WINDOWS), dtype=np.float32)
        for kid in range(args.n_kernels):
            counts = code_counts[kid, codes]
            feat[kid] = raw_lookup[kid, counts] + rng.normal(
                0, args.synthetic_noise, N_WINDOWS)
        feature_rows.append((robust_norm(feat), img))
        images.append(img)
        codes_all.append(codes)
    lookup = fit_lookup_from_features(feature_rows[:args.profile_count], kernel_bits)
    rows = []
    for i in range(args.profile_count, args.n_images):
        img = images[i]
        codes = codes_all[i]
        feat = feature_rows[i][0]
        pred_codes = decode_patches(feat, lookup, code_counts)
        recon = reconstruct_from_codes(pred_codes)
        m = metrics(recon, img)
        m["label"] = int(labels[i])
        m["image_index"] = i
        m["patch_exact"] = float(np.mean(pred_codes == codes))
        rows.append(m)
        np.savez(os.path.join(outdir, f"recovered_{i:04d}.npz"),
                 original=img, recovered=recon, pred_codes=pred_codes, true_codes=codes,
                 label=int(labels[i]))
    summary = summarize(rows)
    with open(os.path.join(outdir, "summary.json"), "w") as fp:
        json.dump({"summary": summary, "images": rows, "mode": "synthetic"}, fp, indent=2)
    print(json.dumps(summary, indent=2))


def summarize(rows):
    keys = [k for k, v in rows[0].items() if isinstance(v, (int, float, np.floating))]
    return {k: float(np.mean([r[k] for r in rows])) for k in keys
            if k not in ("label", "image_index")}


def run_trace_attack(args):
    with open(os.path.join(args.traces, "capture_manifest.json")) as fp:
        manifest = json.load(fp)
    files = sorted(glob.glob(os.path.join(args.traces, "img*.npz")))
    if len(files) <= args.profile_count:
        raise ValueError("need more trace files than --profile-count")
    n_kernels = int(manifest.get("n_kernels", args.n_kernels))
    dwell = int(manifest.get("dwell", args.dwell))
    spc = int(manifest.get("samples_per_cycle", args.samples_per_cycle))
    presamples = int(manifest.get("presamples", 0))
    kernels_path = args.kernels or manifest["kernels"]
    probe_kernels = manifest.get("probe_kernels", args.probe_kernels)
    kernel_bits = load_kernel_bits(kernels_path, n_kernels, probe_kernels)

    profile_files = files[:args.profile_count]
    eval_files = files[args.profile_count:]
    if args.eval_count is not None:
        eval_files = eval_files[:args.eval_count]
    lookup = fit_lookup(profile_files, kernel_bits, dwell, spc, presamples, args.feature)
    code_counts = match_counts_for_codes(kernel_bits)

    os.makedirs(args.out, exist_ok=True)
    rows = []
    for path in eval_files:
        d = np.load(path)
        img = d["image"].astype(np.uint8)
        true_codes = patch_codes_from_patches(patch_matrix_from_image(img))
        feat = extract_features(path, dwell, spc, presamples, args.feature)
        pred_codes = decode_patches(feat, lookup, code_counts)
        recon = reconstruct_from_codes(pred_codes)
        m = metrics(recon, img)
        m["patch_exact"] = float(np.mean(pred_codes == true_codes))
        m["label"] = int(d["label"])
        m["file"] = os.path.basename(path)
        rows.append(m)
        outname = f"recovered_{os.path.basename(path)[3:7]}.npz"
        np.savez(os.path.join(args.out, outname),
                 original=img, recovered=recon, pred_codes=pred_codes,
                 true_codes=true_codes, label=int(d["label"]))
        print(f"{os.path.basename(path)} bit_acc={m['bit_acc']:.4f} "
              f"f1={m['foreground_f1']:.4f} patch={m['patch_exact']:.4f}")

    summary = summarize(rows)
    payload = {
        "summary": summary,
        "images": rows,
        "capture_manifest": manifest,
        "feature": args.feature,
        "profile_count": args.profile_count,
    }
    with open(os.path.join(args.out, "summary.json"), "w") as fp:
        json.dump(payload, fp, indent=2)
    print(json.dumps(summary, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", help="capture directory with img*.npz")
    ap.add_argument("--kernels", default="../training/artifacts/model_3x3/layer1_kernels.npy")
    ap.add_argument("--probe-kernels", choices=["onehot", "file"], default="onehot")
    ap.add_argument("--out", default="results_leakage_reconstruct")
    ap.add_argument("--profile-count", type=int, default=20)
    ap.add_argument("--eval-count", type=int, default=None)
    ap.add_argument("--feature", choices=["mean_abs", "rms", "p2p"], default="mean_abs")
    ap.add_argument("--n-kernels", type=int, default=9)
    ap.add_argument("--dwell", type=int, default=8)
    ap.add_argument("--samples-per-cycle", type=int, default=4)
    ap.add_argument("--synthetic-selftest", action="store_true")
    ap.add_argument("--images", default="../host/mnist_test.npz")
    ap.add_argument("--n-images", type=int, default=40)
    ap.add_argument("--threshold", type=int, default=127)
    ap.add_argument("--synthetic-noise", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    if args.synthetic_selftest:
        synthetic_selftest(args)
    else:
        if not args.traces:
            raise SystemExit("need --traces or --synthetic-selftest")
        run_trace_attack(args)


if __name__ == "__main__":
    main()
