#!/usr/bin/env python3
"""Validate the calibrated simulation against a KNOWN hardware result, before trusting
any RGB forecast made through it. SIMULATION ONLY -- no hardware is touched.

The question
-----------
Phase 1's linear probe could not resolve RGB feasibility at the bench's measured noise.
The fix is to run the actual nonlinear reconstructor -- but a forecast from a simulation
is only worth acting on if the simulation reproduces something already measured.

So: feed the generator SIMULATED traces of the same images, the same deployed kernel and
the same noise level as a real capture whose outcome is known, and see whether it lands
near the measured number.

    hardware: zero-amplifier grey design, Fashion-MNIST, deployed kernel 0, k=2.701,
              avg=1  ->  MSSIM 0.724, recognition 0.746
    (attack/hardtests_20260830/p2p_paper60k_noamp/summary.json)

Like-for-like matters. Fashion-MNIST is far more stereotyped than natural colour images,
so anchoring against CIFAR luminance would compare an easy task to a hard one and make a
sound simulation look broken. This uses the hardware run's own image file and kernel.

Reading the result
------------------
close   -> the simulation is predictive; the RGB forecast can inform bench time.
far     -> the forecast is NOT usable. Report that rather than using the RGB numbers to
           justify or discourage a capture. A simulation that cannot reproduce a measured
           grayscale result has not earned the right to predict an unmeasured RGB one.

Writes results.json incrementally so a long run survives an interruption.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

HW_MSSIM = 0.724
HW_RECOG = 0.746
HW_K = 2.701
FASHION_NPZ = "host/fashion_train.npz"
DEPLOYED_KERNELS = "training/artifacts/model_3x3/layer1_kernels.npy"
PREDICTIVE_GAP = 0.15          # MSSIM gap within which the sim is treated as usable


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-images", type=int, default=20000)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--k", type=float, default=None,
                    help="override the calibrated k (default: measure it)")
    ap.add_argument("--capture", default="host/traces_gfash_noamp_train60k")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--noise-model", choices=["scalar", "sampled"], default="sampled",
                    help="scalar = the old per-window i.i.d. Gaussian model; "
                         "sampled = sample-granularity traces built from real measured "
                         "residuals (default)")
    ap.add_argument("--avg", type=int, default=1,
                    help="averaging factor; only avg=1 is validated by round-trip")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="experiments/rgb_sca/results/anchor_validation")
    args = ap.parse_args()

    import torch
    from torch import nn
    torch.set_num_threads(args.threads)

    from . import rgb_bench as B
    from . import rgb_generator as G
    from . import rgb_noise as NZ
    from .run_phase1_generator import simulate_features, _Args

    if args.k is None:
        calib = B.calibrate_from_capture(args.capture, 1024, "settle_6_13")
        k = calib["k_calibrated"]
        bits = calib["bitstream_sha256"]
    else:
        k, bits = float(args.k), None
    print(f"k = {k:.3f}   hardware anchor: MSSIM {HW_MSSIM}, recognition {HW_RECOG}")

    fash = np.load(FASHION_NPZ)["images"][:args.n_images].astype(np.uint8)[:, None, :, :]
    dep = np.load(DEPLOYED_KERNELS)[0].astype(np.int32)[None, None, :, :]
    print(f"anchor images {fash.shape}, deployed kernel 0")

    if args.noise_model == "sampled":
        ch = NZ.characterize_capture(args.capture, 1500)
        print(f"sample-granularity model: pool {ch['residual'].shape}, "
              f"gain peak at sample {int(np.abs(ch['gain']).argmax())}")
        # Valid-convolution power, matching the capture's own geometry. The default
        # same-padded power would be 784 windows against a 676-window residual pool.
        pw = np.stack([B.grayscale_model_power(im[0], dep[0, 0], ch["out_side"])
                       for im in fash])
        feats = NZ.simulate_features_sampled(fash, dep, "summed", ch,
                                             n_avg=args.avg, seed=args.seed, power=pw)
        sigma = float(ch["power_std"])
        del ch
    else:
        feats, sigma = simulate_features(fash, dep, "summed", k, args.seed)
    g3 = np.repeat(fash, 3, axis=1)
    del fash
    print(f"features {feats.shape}  noise_model={args.noise_model}", flush=True)

    t0 = time.time()
    res, _rec, _truth, _te = G.train_arm(
        feats, g3, "trace", _Args(args.epochs, args.batch, args.lr, args.seed),
        torch, nn, np.random.default_rng(args.seed))
    mssim = float(np.mean(res["mssim_per_channel"]))
    gap = abs(mssim - HW_MSSIM)

    out = {
        "kind": "rgb_sca_anchor_validation",
        "source": "simulation_calibrated_to_hardware",
        "hardware_used": False,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "args": vars(args),
        "k": k,
        "noise_model": args.noise_model,
        "n_avg": args.avg,
        "calibration_bitstream_sha256": bits,
        "hardware_anchor": {"mssim": HW_MSSIM, "recognition": HW_RECOG, "k": HW_K,
                            "dataset": "fashion-mnist, deployed kernel 0, avg=1"},
        "simulated": {"mssim_mean": mssim, **res},
        "abs_gap": gap,
        "simulation_is_predictive": bool(gap < PREDICTIVE_GAP),
        "reading": ("If simulation_is_predictive is false, any RGB forecast made through "
                    "this simulation is not usable evidence about bench feasibility."),
        "elapsed_sec": round(time.time() - t0, 1),
    }

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "results.json"), "w") as fh:
        json.dump(out, fh, indent=1, default=float)

    print(f"\nsimulated MSSIM {mssim:.3f}  vs hardware {HW_MSSIM}  (gap {gap:.3f})")
    print(f"MAE {res['mae_pooled']:.2f}   {out['elapsed_sec']:.0f}s")
    print(f"SIMULATION PREDICTIVE: {out['simulation_is_predictive']}")
    print(f"wrote {os.path.join(args.out, 'results.json')}")


if __name__ == "__main__":
    main()
