#!/usr/bin/env python3
"""Phase 1 driver: does the RGB signal survive THIS bench? SIMULATION, CALIBRATED.

No hardware is touched. Reads an existing capture to measure the bench's noise level,
then replays the Phase 0 probes at that measured level. Records are stamped
source="simulation_calibrated_to_hardware" together with the sha256 of the bitstream
whose capture supplied the calibration, so the number can always be traced to a
specific measurement.

The question Phase 1 answers
---------------------------
Phase 0 showed per-channel information survives the first-layer accumulation at zero
noise. That is an architectural ceiling. Phase 1 asks whether the CW305/CW-Lite bench,
at its MEASURED signal-to-residual ratio, leaves enough of it to work with -- the
separation the feasibility review flagged as unresolved question 2.

The sufficiency anchor
----------------------
The calibration capture is one on which the GRAYSCALE attack demonstrably works
(MSSIM 0.724, recognition 0.746). So the grayscale arm of this run is not a curiosity:
it is the control that converts an abstract "advantage over prior" into a calibrated
yardstick. If an RGB dataflow retains an advantage comparable to the grayscale arm at
the same k, that is meaningful evidence. If it collapses while grayscale holds, the
colour information is being lost specifically.

Usage:
    python -m experiments.rgb_sca.run_phase1 --out experiments/rgb_sca/results/phase1_<date>
    python -m experiments.rgb_sca.run_phase1 --dataset cifar10 --n-images 400
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time

import numpy as np

from . import rgb_bench as B
from . import rgb_forward as F
from . import rgb_probe as P
from .run_phase0 import load_images

RESULT_KIND = "rgb_sca_phase1_calibrated_simulation"
DEFAULT_CAPTURE = "host/traces_gfash_noamp_train60k"


def load_cifar(n: int, size: int, seed: int):
    """CIFAR-10 test split as (N,3,H,W) uint8 -- genuinely independent scenes.

    Chosen because it is the dataset Wei et al. name in Appendix B as the untested
    future target for multi-channel inputs, and because it removes Phase 0's worst
    limitation (4 source photos).
    """
    from tensorflow.keras.datasets import cifar10
    (_, _), (xte, _) = cifar10.load_data()
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(xte), size=min(n, len(xte)), replace=False)
    imgs = xte[idx].transpose(0, 3, 1, 2).astype(np.uint8)
    if size != 32:
        from PIL import Image
        out = []
        for im in imgs:
            p = Image.fromarray(im.transpose(1, 2, 0)).resize((size, size), Image.LANCZOS)
            out.append(np.asarray(p, dtype=np.uint8).transpose(2, 0, 1))
        imgs = np.stack(out)
    return imgs, {"dataset": "cifar10-test", "n_independent_scenes": int(len(imgs)),
                  "augmented": 0, "selection_seed": seed}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--capture", default=DEFAULT_CAPTURE,
                    help="existing capture directory used ONLY to measure noise")
    ap.add_argument("--dataset", choices=["cifar10", "assets"], default="cifar10")
    ap.add_argument("--n-images", type=int, default=300)
    ap.add_argument("--size", type=int, default=32, choices=[32, 64, 128])
    ap.add_argument("--kernels", type=int, default=9)
    ap.add_argument("--avg", type=int, nargs="*", default=[1, 10, 50, 100, 500],
                    help="averaging levels; k_eff = k / sqrt(avg)")
    ap.add_argument("--feature", default="settle_6_13", choices=list(B.FEATURE_WINDOWS))
    ap.add_argument("--calib-traces", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    t0 = time.time()

    print("STEP 1 - calibrate against a real capture (read-only)")
    calib = B.calibrate_from_capture(args.capture, args.calib_traces, args.feature)
    k = calib["k_calibrated"]
    print(f"  {os.path.basename(args.capture)}  bitstream {calib['bitstream_sha256'][:16]}")
    print(f"  mode={calib['capture_mode']} avg={calib['avg']} input={calib['input_mode']}")
    for name, v in calib["per_feature"].items():
        print(f"    {name:12s} corr {v['correlation']:+.4f} -> k {v['k_noise_over_signal']:.2f}")
    sm = calib["smear"]
    print(f"  smear excess beyond window overlap: {sm['excess_attributable_to_smear']:+.4f}")
    print(f"  -> calibrated k = {k:.3f} (feature={args.feature})")

    print("\nSTEP 2 - load scenes")
    if args.dataset == "cifar10":
        images, prov = load_cifar(args.n_images, args.size, args.seed)
    else:
        images, prov = load_images(args.size, None, max(0, args.n_images - 4), args.seed)
    print(f"  {images.shape}  independent scenes: {prov['n_independent_scenes']}")

    # Grayscale arm: luminance of the same scenes, same probe, same k. This is the
    # sufficiency control -- the bench is known to support grayscale recovery at this k.
    grey = (0.299 * images[:, 0] + 0.587 * images[:, 1] + 0.114 * images[:, 2])
    grey = grey.astype(np.uint8)[:, None, :, :]

    kernels_rgb = F.random_kernels(args.kernels, C=3, K=3, seed=args.seed + 1)
    kernels_grey = F.random_kernels(args.kernels, C=1, K=3, seed=args.seed + 1)

    out = {
        "kind": RESULT_KIND,
        "source": "simulation_calibrated_to_hardware",
        "hardware_used": False,
        "hardware_read_only_calibration": args.capture,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "args": vars(args),
        "provenance": prov,
        "calibration": calib,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "grayscale_reference": {},
        "rgb": {},
    }

    print("\nSTEP 3 - grayscale reference arm (sufficiency control)")
    print("   the bench is KNOWN to support grayscale recovery at avg=1 on this capture")
    for n_avg in args.avg:
        ke = B.k_after_averaging(k, n_avg)
        r = P.recovery_probe(grey, kernels_grey, "summed", noise=ke, seed=args.seed)
        out["grayscale_reference"][f"avg_{n_avg}"] = {"k_eff": ke, **r}
        print(f"   avg={n_avg:<4d} k={ke:6.3f}  advantage {r['advantage_over_prior_pooled']:+6.2f}"
              f"   (trace {r['trace']['mae_pooled']:.2f} vs prior {r['prior_only']['mae_pooled']:.2f})")

    print("\nSTEP 4 - RGB arms at the same measured k")
    for df in F.DATAFLOWS:
        out["rgb"][df] = {}
        print(f"\n  {df}")
        for n_avg in args.avg:
            ke = B.k_after_averaging(k, n_avg)
            r = P.recovery_probe(images, kernels_rgb, df, noise=ke, seed=args.seed)
            out["rgb"][df][f"avg_{n_avg}"] = {"k_eff": ke, **r}
            adv = r["advantage_over_prior_per_channel"]
            print(f"   avg={n_avg:<4d} k={ke:6.3f}  advantage pooled "
                  f"{r['advantage_over_prior_pooled']:+6.2f}  per-ch "
                  f"R{adv[0]:+5.2f} G{adv[1]:+5.2f} B{adv[2]:+5.2f}")

    out["elapsed_sec"] = round(time.time() - t0, 2)
    if args.out:
        os.makedirs(args.out, exist_ok=True)
        path = os.path.join(args.out, "results.json")
        with open(path, "w") as fh:
            json.dump(out, fh, indent=1, default=float)
        print(f"\nwrote {path}")
    return out


if __name__ == "__main__":
    main()
