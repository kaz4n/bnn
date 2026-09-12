#!/usr/bin/env python3
"""Phase 1b: run the 3-channel generator on SIMULATED traces at the measured bench noise.

Purpose: decide how much averaging a real capture needs BEFORE spending bench time, and
decide whether an RGB capture is worth running at all.

SIMULATION, calibrated. No hardware is touched.

The grayscale anchor is what makes this readable
------------------------------------------------
Phase 1's linear probe could not resolve the question at bench noise -- the grayscale
control scored only +0.45 MAE advantage while the real attack on that same capture
reaches MSSIM 0.724 and recognition 0.746. So the probe under-read, and the fix is to run
the actual nonlinear reconstructor the published attacks use.

That only helps if the simulation is calibrated. This driver therefore runs a GRAYSCALE
arm at the same k as the RGB arms, and its result is checked against the known hardware
outcome:

    hardware, k=2.70, avg=1, zero-amplifier grey design -> MSSIM 0.724, recog 0.746

If the simulated grayscale arm lands near that, the simulation is predictive and the RGB
numbers carry weight. If it lands far away, the RGB prediction is NOT trustworthy and
should be reported as such rather than used to justify or discourage bench time. That
check is printed explicitly and stored in the result.

Noise model
-----------
k is defined as sigma_noise / sigma_signal in PER-POSITION CENTRED feature units -- the
same units it was measured in (rgb_bench.calibrate_from_capture). So the data-dependent
signal is isolated by subtracting the per-position mean ACROSS IMAGES, not the mean across
positions within an image. Getting this wrong inflates sigma_signal by the fixed
per-position structure, which is data-independent and cancels in the real measurement.
Averaging N captures divides sigma_noise by sqrt(N).
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from . import rgb_bench as B
from . import rgb_forward as F
from . import rgb_generator as G
from . import rgb_noise as NZ
from .run_phase1 import load_cifar

RESULT_KIND = "rgb_sca_phase1b_generator_on_simulated_traces"
DEFAULT_CAPTURE = "host/traces_gfash_noamp_train60k"

# Known hardware outcome at the calibration point, used to validate the simulation.
#
# The anchor must be LIKE FOR LIKE. The hardware number was obtained on Fashion-MNIST
# with the deployed 3x3 kernel, and Fashion-MNIST is far more stereotyped than natural
# colour images -- centred objects on a black ground, ten classes. Anchoring against
# CIFAR luminance instead would compare a hard reconstruction task against an easy one
# and make the simulation look broken when it was merely being asked something harder.
# So the anchor arm uses the SAME images and the SAME kernel as the hardware run.
HW_ANCHOR = {"mssim": 0.724, "recognition": 0.746, "k": 2.701, "avg": 1,
             "dataset": "fashion-mnist 28x28, deployed kernel index 0",
             "source": "attack/hardtests_20260830/p2p_paper60k_noamp/summary.json"}
FASHION_NPZ = "host/fashion_train.npz"
DEPLOYED_KERNELS = "training/artifacts/model_3x3/layer1_kernels.npy"


def simulate_features(images, kernels, dataflow, k_eff, seed):
    """Noise-free power features plus calibrated noise, in the units k was measured in."""
    clean = np.stack([F.power_features(im, kernels, dataflow).ravel() for im in images])
    clean = clean.astype(np.float64)
    # Isolate the data-dependent part exactly as the calibration did: remove the
    # per-position mean across images. The removed component is data-independent and
    # cancels in the real measurement, so including it would overstate the signal.
    centred = clean - clean.mean(axis=0, keepdims=True)
    sigma_signal = float(centred.std()) or 1.0
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, k_eff * sigma_signal, size=clean.shape)
    return (clean + noise).astype(np.float32), sigma_signal


class _Args:
    """Training hyperparameters, matching Power2Picture where they are stated."""
    def __init__(self, epochs, batch, lr, seed, test_frac=0.1, loss="gradmse"):
        self.epochs, self.batch, self.lr = epochs, batch, lr
        self.seed, self.test_frac, self.loss = seed, test_frac, loss


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--capture", default=DEFAULT_CAPTURE)
    ap.add_argument("--n-images", type=int, default=1500)
    ap.add_argument("--size", type=int, default=32)
    ap.add_argument("--dataflows", nargs="*", default=["summed"],
                    choices=list(F.DATAFLOWS))
    ap.add_argument("--avg", type=int, nargs="*", default=[1, 10, 50, 200])
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--kernels", type=int, default=1,
                    help="1 matches the P2P-style single deployed kernel the "
                         "calibration capture used")
    ap.add_argument("--noise-model", choices=["scalar", "sampled"], default="sampled",
                    help="sampled (default) builds sample-granularity traces from real "
                         "measured residuals; scalar is the superseded per-window i.i.d. "
                         "Gaussian model, kept only for comparison")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip-grey", action="store_true",
                    help="skip the anchor arm (not recommended: it is the validation)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import torch
    from torch import nn

    t0 = time.time()
    calib = B.calibrate_from_capture(args.capture, 1024, "settle_6_13")
    k = calib["k_calibrated"]
    print(f"calibrated k = {k:.3f} from {os.path.basename(args.capture)} "
          f"(bitstream {calib['bitstream_sha256'][:12]})")

    images, prov = load_cifar(args.n_images, args.size, args.seed)
    print(f"RGB images {images.shape}  independent scenes {prov['n_independent_scenes']}")

    # Anchor arm: the same Fashion-MNIST images and the same deployed kernel the hardware
    # run used, so the simulated MSSIM is directly comparable to the measured 0.724.
    fash = np.load(FASHION_NPZ)["images"][:args.n_images].astype(np.uint8)[:, None, :, :]
    dep = np.load(DEPLOYED_KERNELS)[0].astype(np.int32)[None, None, :, :]
    print(f"anchor images {fash.shape} (fashion-mnist, deployed kernel)")

    k_rgb = F.random_kernels(args.kernels, C=3, K=3, seed=args.seed + 1)
    k_grey = dep

    ch = None
    if args.noise_model == "sampled":
        ch = NZ.characterize_capture(args.capture, 1500)
        print(f"noise model: sample-granularity, pool {ch['residual'].shape}, "
              f"gain peak at sample {int(np.abs(ch['gain']).argmax())}")
        print("  NOTE: the residual pool is from the grey 28x28 capture; simulating the "
              f"RGB geometry tiles it across windows (pool {ch['n_windows_pool']}).")
    else:
        print("noise model: scalar i.i.d. Gaussian (SUPERSEDED -- measurably pessimistic)")

    def _feats(imgs, kerns, dataflow, n_avg, k_eff):
        if ch is not None:
            return NZ.simulate_features_sampled(imgs, kerns, dataflow, ch,
                                                n_avg=n_avg, seed=args.seed)
        return simulate_features(imgs, kerns, dataflow, k_eff, args.seed)[0]
    targs = _Args(args.epochs, args.batch, args.lr, args.seed)

    out = {
        "kind": RESULT_KIND,
        "source": "simulation_calibrated_to_hardware",
        "hardware_used": False,
        "hardware_read_only_calibration": args.capture,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "args": vars(args), "provenance": prov,
        "noise_model": args.noise_model,
        "calibration": {kk: calib[kk] for kk in
                        ("k_calibrated", "bitstream_sha256", "capture_mode", "avg")},
        "hardware_anchor": HW_ANCHOR,
        "grayscale_anchor": {}, "rgb": {},
    }

    if not args.skip_grey:
        print("\nGRAYSCALE ANCHOR -- validates the simulation against a known hardware result")
        print(f"  hardware at k={HW_ANCHOR['k']}, avg={HW_ANCHOR['avg']}: "
              f"MSSIM {HW_ANCHOR['mssim']}, recognition {HW_ANCHOR['recognition']}")
        for n_avg in args.avg:
            ke = B.k_after_averaging(k, n_avg)
            feats = _feats(fash, k_grey, "summed", n_avg, ke)
            # The grey arm reuses the RGB generator by replicating its single channel,
            # so the model capacity is identical and the comparison is like for like.
            g3 = np.repeat(fash, 3, axis=1)
            res, _ = G.run_all_arms(feats, g3, targs, torch, nn)
            mssim = float(np.mean(res["trace"]["mssim_per_channel"]))
            out["grayscale_anchor"][f"avg_{n_avg}"] = {
                "k_eff": ke, "mssim_mean": mssim, **res}
            print(f"  avg={n_avg:<4d} k={ke:6.3f}  MSSIM {mssim:.3f}"
                  f"  trace MAE {res['trace']['mae_pooled']:6.2f}"
                  f"  prior {res['prior_only']['mae_pooled']:6.2f}"
                  f"  advantage {res['summary']['advantage_over_prior_pooled_mae']:+6.2f}")

        # Validate against the MATCHED condition only. An earlier version searched every
        # averaging level for whichever happened to land closest to the anchor -- but the
        # anchor is an avg=1 measurement, so a badly mismatched avg=1 simulation could be
        # declared predictive because avg=50 coincidentally matched. Compare like with
        # like, and fail loudly when the matching condition is absent.
        # Found in external review, 12 September 2026.
        anchor_key = f"avg_{HW_ANCHOR['avg']}"
        if anchor_key not in out["grayscale_anchor"]:
            raise SystemExit(
                f"anchor condition {anchor_key} was not run, so the simulation cannot be "
                f"validated. The hardware anchor is avg={HW_ANCHOR['avg']}; include it in "
                "--avg rather than substituting another averaging level.")
        matched = out["grayscale_anchor"][anchor_key]
        out["anchor_check"] = {
            "matched_condition": anchor_key,
            "k_eff": matched["k_eff"],
            "sim_mssim": matched["mssim_mean"],
            "hw_mssim": HW_ANCHOR["mssim"],
            "abs_gap": abs(matched["mssim_mean"] - HW_ANCHOR["mssim"]),
            "simulation_is_predictive":
                abs(matched["mssim_mean"] - HW_ANCHOR["mssim"]) < 0.15,
            "reading": ("Compared at the anchor's own averaging level. Matching averaging "
                        "is necessary but not sufficient: dataset, split, kernel, training "
                        "budget and metric definition must also correspond."),
        }
        ac = out["anchor_check"]
        print(f"\n  ANCHOR CHECK: closest simulated MSSIM {ac['sim_mssim']:.3f} vs "
              f"hardware {ac['hw_mssim']:.3f} (gap {ac['abs_gap']:.3f}) -> "
              f"predictive: {ac['simulation_is_predictive']}")

    for df in args.dataflows:
        out["rgb"][df] = {}
        print(f"\nRGB dataflow: {df}")
        for n_avg in args.avg:
            ke = B.k_after_averaging(k, n_avg)
            feats = _feats(images, k_rgb, df, n_avg, ke)
            res, _ = G.run_all_arms(feats, images, targs, torch, nn)
            out["rgb"][df][f"avg_{n_avg}"] = {"k_eff": ke, **res}
            t, s = res["trace"], res["summary"]
            cp = t["channel_permutation"]
            print(f"  avg={n_avg:<4d} k={ke:6.3f}"
                  f"  MAE/ch {[round(v,1) for v in t['mae_per_channel']]}"
                  f"  luma {t['luma_mae']:5.2f} chroma {[round(v,2) for v in t['chroma_mae']]}")
            print(f"{'':14s}advantage {s['advantage_over_prior_pooled_mae']:+6.2f}"
                  f"  luma {s['luma_advantage']:+5.2f}"
                  f"  chroma {[round(v,2) for v in s['chroma_advantage']]}"
                  f"  swap-penalty {cp['swap_penalty']:+.2f}"
                  f"  shuffled~prior {s['shuffled_matches_prior']}")

    out["elapsed_sec"] = round(time.time() - t0, 2)
    if args.out:
        os.makedirs(args.out, exist_ok=True)
        with open(os.path.join(args.out, "results.json"), "w") as fh:
            json.dump(out, fh, indent=1, default=float)
        print(f"\nwrote {os.path.join(args.out, 'results.json')}")


if __name__ == "__main__":
    main()
