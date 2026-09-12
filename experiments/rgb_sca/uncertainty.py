#!/usr/bin/env python3
"""Uncertainty estimates by rescoring FROZEN predictions. No retraining, no hardware.

Experiment 2 from the external review. Every figure reported so far is a point estimate
over a pooled test set, so nothing could be said about whether a difference between two
conditions is real. This computes confidence intervals, and it does so at the right unit.

THE INDEPENDENT UNIT IS THE IMAGE, NOT THE PIXEL
------------------------------------------------
A 32x32 RGB image has 3,072 values, but they are not 3,072 independent observations of
the method: neighbouring pixels are correlated, and all of them share one trace, one
scene, and one capture. Treating pixels as independent would shrink every interval by
roughly sqrt(3072) and manufacture significance. So per-image errors are computed first,
and all resampling is over IMAGES.

PAIRED WHEREVER POSSIBLE
------------------------
Two conditions evaluated on the SAME test images are compared by their per-image
difference, not by the difference of their means. Pairing removes the between-image
variance -- which dominates here, since some images are far easier than others -- and is
the difference between a usable interval and a useless one.

The train/test split is seeded identically across dataflows, so the test sets match and
cross-dataflow comparisons can be paired too. That is checked, not assumed.
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np

R = "experiments/rgb_sca/results"


def per_image_mae(a, b):
    """MAE per image in 0-255 units. Shape (N,3,H,W) -> (N,)."""
    return np.abs(np.clip(a, 0, 1) - b).reshape(len(a), -1).mean(axis=1) * 255.0


def per_image_chroma_mae(a, b):
    """Mean |Cb| + |Cr| error per image, on the same 0-255 footing as luma."""
    def ycc(x):
        r, g, bl = x[:, 0], x[:, 1], x[:, 2]
        y = 0.299 * r + 0.587 * g + 0.114 * bl
        return 0.564 * (bl - y), 0.713 * (r - y)
    acb, acr = ycc(np.clip(a, 0, 1))
    bcb, bcr = ycc(b)
    n = len(a)
    return ((np.abs(acb - bcb).reshape(n, -1).mean(1)
             + np.abs(acr - bcr).reshape(n, -1).mean(1)) / 2.0) * 255.0


def boot_ci(x, n_boot=10000, alpha=0.05, seed=0):
    """Percentile bootstrap over the sample (images), returning mean and CI."""
    x = np.asarray(x, dtype=np.float64)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    means = x[idx].mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"mean": float(x.mean()), "ci_low": float(lo), "ci_high": float(hi),
            "n": int(len(x)),
            "excludes_zero": bool(lo > 0 or hi < 0)}


def load_cell(mode, group):
    base = (f"{R}/hardware_parallel_rev3" if mode == "parallel"
            else f"{R}/hardware_20260912_rev3")
    f = f"{base}/images/{mode}/reconstructions_{group}.npz"
    if not os.path.exists(f):
        return None
    z = np.load(f)
    return {k: z[k] for k in z.files}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=f"{R}/uncertainty")
    args = ap.parse_args()

    modes = ("summed", "parallel", "serial")
    groups = ("control", "natural")
    out = {"kind": "rgb_sca_uncertainty_rescoring", "source": "rescored_frozen_predictions",
           "hardware_used": False, "n_bootstrap": args.boot,
           "unit_of_analysis": "image", "cells": {}, "paired_dataflow": {}}

    print("Per-image paired advantage over the analytic prior (MAE, 0-255).")
    print("Positive = reconstruction better than the prior. CI is a 10k percentile")
    print("bootstrap over IMAGES, not pixels.\n")
    print(f"{'mode':10s} {'group':9s} {'advantage':>10s} {'95% CI':>20s}  {'chroma adv':>11s} {'95% CI':>20s}")

    cells = {}
    for m in modes:
        for g in groups:
            c = load_cell(m, g)
            if c is None:
                continue
            adv = per_image_mae(c["prior_only"], c["truth"]) - per_image_mae(c["trace"], c["truth"])
            cadv = (per_image_chroma_mae(c["prior_only"], c["truth"])
                    - per_image_chroma_mae(c["trace"], c["truth"]))
            ci, cci = boot_ci(adv, args.boot, seed=args.seed), boot_ci(cadv, args.boot, seed=args.seed)
            cells[(m, g)] = {"adv": adv, "cadv": cadv, "idx": c["test_indices"]}
            out["cells"][f"{m}/{g}"] = {"advantage": ci, "chroma_advantage": cci}
            print(f"{m:10s} {g:9s} {ci['mean']:+10.2f} [{ci['ci_low']:+7.2f},{ci['ci_high']:+7.2f}]"
                  f"  {cci['mean']:+11.2f} [{cci['ci_low']:+7.2f},{cci['ci_high']:+7.2f}]")

    # ---- paired cross-dataflow comparison, only where the test sets actually match
    print("\nPaired dataflow differences (same images, per-image difference).")
    print("A CI spanning zero means the ordering is not resolved by this data.\n")
    for g in groups:
        for a, b in (("summed", "serial"), ("summed", "parallel"), ("parallel", "serial")):
            ca, cb = cells.get((a, g)), cells.get((b, g))
            if ca is None or cb is None:
                continue
            if not np.array_equal(ca["idx"], cb["idx"]):
                print(f"  {g:9s} {a} vs {b}: test sets differ -- not paired, skipped")
                continue
            d = ca["adv"] - cb["adv"]
            ci = boot_ci(d, args.boot, seed=args.seed)
            out["paired_dataflow"][f"{g}/{a}_minus_{b}"] = ci
            verdict = "RESOLVED" if ci["excludes_zero"] else "not resolved"
            print(f"  {g:9s} {a:9s} - {b:9s} = {ci['mean']:+6.2f} "
                  f"[{ci['ci_low']:+6.2f},{ci['ci_high']:+6.2f}]   {verdict}")

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "results.json"), "w") as fh:
        json.dump(out, fh, indent=1, default=float)
    print(f"\nwrote {os.path.join(args.out, 'results.json')}")


if __name__ == "__main__":
    main()
