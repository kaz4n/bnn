#!/usr/bin/env python3
"""Original vs recovered montages for any reconstruction route in this project."""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from score_all import load_active, load_npz_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=["active", "bgdetect", "p2p"], required=True)
    ap.add_argument("--results", required=True)
    ap.add_argument("--truth", help="truth json (active only)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--count", type=int, default=10)
    ap.add_argument("--title", default=None)
    args = ap.parse_args()

    if args.kind == "active":
        rec, gt, labels = load_active(args.results, args.truth)
    elif args.kind == "bgdetect":
        rec, gt, labels = load_npz_dir(args.results, "recovered.npz")
    else:
        rec, gt, labels = load_npz_dir(args.results, "test_recovered.npz")

    n = min(args.count, len(rec))
    fig, axes = plt.subplots(2, n, figsize=(1.15 * n, 2.8))
    for i in range(n):
        axes[0, i].imshow(gt[i], cmap="gray", vmin=0, vmax=1)
        axes[0, i].set_title(str(int(labels[i])), fontsize=8)
        axes[1, i].imshow(rec[i], cmap="gray", vmin=0, vmax=1)
        for r in (0, 1):
            axes[r, i].set_aspect("equal")
            axes[r, i].axis("off")
    axes[0, 0].set_ylabel("original")
    axes[1, 0].set_ylabel("recovered")
    fig.suptitle(args.title or f"{args.kind}: original (top) vs recovered (bottom)",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fig.savefig(args.out, dpi=180)
    print(os.path.abspath(args.out))


if __name__ == "__main__":
    main()
