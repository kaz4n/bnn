#!/usr/bin/env python3
"""Train the 3-channel generator on REAL measured CW305 traces and produce reconstructions.

This is the deliverable: recovered RGB images from physically measured side-channel data,
with the controls that decide whether the recovery is real.

GROUPS ARE TRAINED SEPARATELY, deliberately
-------------------------------------------
  natural   Real CIFAR-10 photographs. The actual goal.
  control   Tinted Fashion-MNIST at anchor difficulty. The POSITIVE control -- simulation
            says luminance recovers here, so if the natural group fails this tells you
            whether the capture and pipeline work at all. Without it a null on natural
            images is uninterpretable.
  diag      Solid patches and colour bars, excluded from training and reconstructed only
            as a wiring check.

Pooling the groups would let the easy group carry the hard one and produce a headline
number that describes neither.

Every group reports the full control set: prior-only (analytic), shuffled, the
channel-permutation penalty, and per-channel plus luma/chroma errors.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trace-dir", default="host/traces_rgb_full")
    ap.add_argument("--image-set", default="host/rgb_capture_images.npz")
    ap.add_argument("--modes", nargs="*", default=["summed", "serial"])
    ap.add_argument("--groups", nargs="*", default=["control", "natural"])
    ap.add_argument("--feature", default="settle")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="experiments/rgb_sca/results/hardware_20260912")
    args = ap.parse_args()

    import torch
    from torch import nn
    torch.set_num_threads(args.threads)
    from . import rgb_generator as G
    from .run_phase1_generator import _Args

    grp_all = np.load(args.image_set)["group"]
    targs = _Args(args.epochs, args.batch, args.lr, args.seed)

    out = {
        "kind": "cw305_rgb_hardware_reconstruction",
        "source": "hardware",
        "hardware_used": True,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "args": vars(args), "modes": {},
    }

    for mode in args.modes:
        feats, images, idx, man = G.load_rgb_shards(args.trace_dir, mode,
                                                    feature=args.feature)
        out.setdefault("capture", {
            "bitstream_sha256": man.get("bitstream_sha256"),
            "design_signature": man.get("design_signature"),
            "capture_mode": man.get("capture_mode"),
            "avg": man.get("avg"), "dwell": man.get("dwell"),
            "samples_per_cycle": man.get("samples_per_cycle"),
            "gain_db": man.get("gain_db"), "fpga_freq_hz": man.get("fpga_freq_hz"),
            "kernel_index": man.get("kernel_index"),
            "functional_checks": {m: man["modes"][m]["functional_check"]["mismatches"]
                                  for m in man.get("modes", {})},
        })
        grp = grp_all[idx]
        print(f"\n=== mode {mode}: {feats.shape[0]} traces, {feats.shape[1]} features ===")
        out["modes"][mode] = {}

        for g in args.groups:
            sel = grp == g
            if sel.sum() < 50:
                print(f"  {g}: only {sel.sum()} traces, skipping")
                continue
            print(f"\n  --- group {g}: {int(sel.sum())} traces ---")
            res, recs = G.run_all_arms(feats[sel], images[sel], targs, torch, nn)
            saved = G.save_reconstructions(
                recs, os.path.join(args.out, "images", mode), g, n_show=12, seed=args.seed)
            res["images"] = saved
            out["modes"][mode][g] = res

            t, p, s = res["trace"], res["prior_only"], res["summary"]
            cp = t["channel_permutation"]
            print(f"    trace MAE/ch {[round(v,2) for v in t['mae_per_channel']]} "
                  f"pooled {t['mae_pooled']:.2f}")
            print(f"    prior MAE/ch {[round(v,2) for v in p['mae_per_channel']]} "
                  f"pooled {p['mae_pooled']:.2f}")
            print(f"    shuffled {res['shuffled']['mae_pooled']:.2f}  "
                  f"matches prior: {s['shuffled_matches_prior']}")
            print(f"    ADVANTAGE {s['advantage_over_prior_pooled_mae']:+.2f}   "
                  f"LUMA {s['luma_advantage']:+.2f}   "
                  f"CHROMA {[round(v,2) for v in s['chroma_advantage']]}")
            print(f"    swap penalty {cp['swap_penalty']:+.3f} "
                  f"(ratio {cp['swap_penalty_ratio']:.4f})")
            print(f"    MSSIM/ch {[round(v,3) for v in t['mssim_per_channel']]}")
            print(f"    images: {saved['grid']}", flush=True)

        del feats, images

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "results.json"), "w") as fh:
        json.dump(out, fh, indent=1, default=float)
    print(f"\nwrote {os.path.join(args.out, 'results.json')}")


if __name__ == "__main__":
    main()
