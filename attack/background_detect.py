#!/usr/bin/env python3
"""Wei et al. Section 6 style background detection on the CW305 leakage core.

The reference paper's first attack needs no power template and no profiled
pixel values: it thresholds the per-cycle power magnitude, marks the pixels
processed during low-power cycles as background, and recovers a black/white
image from those markers.

Here one convolution window is one dwell period, so the per-window power
magnitude plays the role of Wei's per-cycle power.  For a one-hot probe kernel
the all-background patch gives the *highest* XNOR match count, so the sign of
the comparison is inverted relative to the paper; the mechanism (one scalar
threshold, no per-pixel profiling) is the same.

A window declared "all background" implies all nine of its pixels are
background, so background markers are unioned over the overlapping windows,
exactly the relation the paper describes for its K*(K+1) window.

Only one scalar (the threshold) is taken from the profiling images.
"""
import argparse
import glob
import json
import os

import numpy as np

from leakage_first_reconstruct import (
    LINE,
    OUT_SIDE,
    N_WINDOWS,
    extract_features,
    metrics,
)


def read_capture(trace_dir):
    return json.load(open(os.path.join(trace_dir, "capture_manifest.json")))


def window_features(path, cap, kernel_id, feature, combine="single"):
    """One scalar per convolution window.

    ``single`` uses one probe kernel, like the single-kernel background
    detection of Wei et al.  ``mean`` averages the per-window magnitude over all
    probe kernels: for one-hot kernels the mean match count is 8 - (7/9)*s in
    the number of foreground pixels s, so the all-background window sits at one
    end of the scale.  Both variants still take only a single scalar threshold
    from the profiling images.
    """
    feat = extract_features(path, int(cap["dwell"]), int(cap["samples_per_cycle"]),
                            int(cap.get("presamples", 0)), feature)
    if combine == "mean":
        return feat.mean(axis=0)
    return feat[kernel_id]


def image_from_background(is_bg_window):
    """Union of all-background windows -> background mask -> binary image."""
    bg = np.zeros((LINE, LINE), dtype=bool)
    idx = 0
    for y in range(OUT_SIDE):
        for x in range(OUT_SIDE):
            if is_bg_window[idx]:
                bg[y:y + 3, x:x + 3] = True
            idx += 1
    img = (~bg).astype(np.uint8)
    # pixels never covered by any window (the outer frame) stay background
    covered = np.zeros((LINE, LINE), dtype=bool)
    covered[0:OUT_SIDE + 2, 0:OUT_SIDE + 2] = True
    img[~covered] = 0
    return img


def load_set(files, cap, kernel_id, feature, combine="single"):
    """Extract window features and ground truth once; threshold search is cheap."""
    feats, truth, labels = [], [], []
    for path in files:
        d = np.load(path)
        feats.append(window_features(path, cap, kernel_id, feature, combine))
        truth.append(d["image"].astype(np.uint8))
        labels.append(int(d["label"]))
    return np.array(feats), np.array(truth), np.array(labels)


def evaluate(feats, truth, threshold, direction):
    accs, f1s, recon = [], [], []
    for feat, gt in zip(feats, truth):
        is_bg = feat >= threshold if direction == "high" else feat <= threshold
        img = image_from_background(is_bg)
        m = metrics(img, gt)
        accs.append(m["bit_acc"])
        f1s.append(m["foreground_f1"])
        recon.append(img)
    return ({"bit_acc": float(np.mean(accs)),
             "foreground_f1": float(np.mean(f1s))}, np.array(recon))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--kernel-id", type=int, default=0)
    ap.add_argument("--feature", choices=["mean_abs", "rms", "p2p"], default="mean_abs")
    ap.add_argument("--profile-count", type=int, default=30,
                    help="images used only to pick the scalar threshold")
    ap.add_argument("--eval-count", type=int, default=30)
    ap.add_argument("--direction", choices=["high", "low", "auto"], default="auto")
    ap.add_argument("--grid", type=int, default=81)
    ap.add_argument("--combine", choices=["single", "mean"], default="single",
                    help="single probe kernel (as in the paper) or mean over all kernels")
    args = ap.parse_args()

    cap = read_capture(args.traces)
    files = sorted(glob.glob(os.path.join(args.traces, "img*.npz")))
    prof = files[:args.profile_count]
    ev = files[args.profile_count:args.profile_count + args.eval_count]
    if not prof or not ev:
        raise SystemExit("need both profile and eval files")

    prof_feats, prof_truth, _ = load_set(prof, cap, args.kernel_id, args.feature,
                                        args.combine)
    ev_feats, truth, labels = load_set(ev, cap, args.kernel_id, args.feature,
                                       args.combine)
    lo = float(np.percentile(prof_feats, 1))
    hi = float(np.percentile(prof_feats, 99))
    grid = np.linspace(lo, hi, args.grid)

    directions = ["high", "low"] if args.direction == "auto" else [args.direction]
    best = None
    for direction in directions:
        for t in grid:
            m, _ = evaluate(prof_feats, prof_truth, float(t), direction)
            if best is None or m["foreground_f1"] > best[0]:
                best = (m["foreground_f1"], float(t), direction, m)
    _, threshold, direction, prof_metrics = best

    ev_metrics, recon = evaluate(ev_feats, truth, threshold, direction)

    try:
        from recognize_numpy import GoldenMLP

        clf = GoldenMLP()
        ev_metrics["recognition_accuracy_original"] = float(
            np.mean(clf.predict(truth.astype(np.float32)) == labels))
        ev_metrics["recognition_accuracy_recovered"] = float(
            np.mean(clf.predict(recon.astype(np.float32)) == labels))
    except Exception as exc:
        ev_metrics["recognition_error"] = str(exc)

    os.makedirs(args.out, exist_ok=True)
    np.savez_compressed(os.path.join(args.out, "recovered.npz"),
                        recovered=recon, truth=truth, labels=labels)
    payload = {
        "kind": "cw305_background_detection",
        "trace_dir": os.path.abspath(args.traces),
        "capture_manifest": cap,
        "kernel_id": args.kernel_id,
        "combine": args.combine,
        "feature": args.feature,
        "threshold": threshold,
        "direction": direction,
        "profile_metrics": prof_metrics,
        "eval_metrics": ev_metrics,
        "profile_count": len(prof),
        "eval_count": len(ev),
    }
    with open(os.path.join(args.out, "summary.json"), "w") as fp:
        json.dump(payload, fp, indent=2)
    print(json.dumps({"threshold": threshold, "direction": direction,
                      "eval": ev_metrics}, indent=2))


if __name__ == "__main__":
    main()
