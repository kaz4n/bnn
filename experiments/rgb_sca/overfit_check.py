#!/usr/bin/env python3
"""Can the training pipeline fit a tiny, correctly-paired problem at all?

Experiment 3 from the external review. The natural-image group scores worse than the
analytic prior, and that has (at least) four possible causes:

    implementation defect / mispairing / bad scaling   -> cannot fit ANYTHING
    insufficient optimization                          -> cannot fit even 8 images
    poor generalization                                -> fits 8, fails held-out
    input carries no usable information                -> fits 8, fails held-out

The first two are excluded by showing the pipeline CAN memorise a handful of images; the
last two are not distinguished by this test and we do not claim otherwise. Memorising
eight images is a functionality check, not evidence of recovery.

THREE ARMS, and the synthetic one is the load-bearing control:

  synthetic   Features are a fixed random linear projection of the image itself, plus a
              little noise. The mapping is invertible in principle, so a working pipeline
              MUST fit it. If this arm fails, nothing downstream is interpretable and the
              defect is in the model, the loss, the scaling or the pairing -- not in the
              side channel.
  natural     8 real CIFAR traces. The failing case.
  control     8 real tinted-garment traces. The working case, for calibration.

Reported per arm: final training MAE, the loss trajectory, the gradient norm, and the
across-sample standard deviation of the predictions. That last one matters -- a model
that has collapsed to emitting the dataset mean regardless of input has near-zero
prediction spread, which looks like "trained" in the loss but is not fitting anything.

Budget is predeclared (see --epochs) and identical across arms.
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
    ap.add_argument("--mode", default="summed")
    ap.add_argument("--image-set", default="host/rgb_capture_images.npz")
    ap.add_argument("--n", type=int, default=8, help="images to memorise")
    ap.add_argument("--epochs", type=int, default=400, help="predeclared budget")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="experiments/rgb_sca/results/overfit_check")
    args = ap.parse_args()

    import torch
    from torch import nn
    torch.set_num_threads(args.threads)
    from . import rgb_generator as G

    grp_all = np.load(args.image_set)["group"]
    feats, images, idx, _man = G.load_rgb_shards(args.trace_dir, args.mode)
    grp = grp_all[idx]

    arms = {}
    for g in ("control", "natural"):
        sel = np.where(grp == g)[0][:args.n]
        arms[g] = (feats[sel].astype(np.float32), images[sel])

    # Synthetic arm: features ARE the image under a fixed random projection, so the
    # mapping is learnable by construction. Same shapes, same scale as the real arms.
    rng = np.random.default_rng(args.seed)
    imgs = arms["natural"][1]
    flat = imgs.reshape(len(imgs), -1).astype(np.float64) / 255.0
    proj = rng.normal(0, 1, size=(flat.shape[1], arms["natural"][0].shape[1]))
    syn = flat @ proj / np.sqrt(flat.shape[1])
    syn = syn + rng.normal(0, 0.01 * syn.std(), size=syn.shape)
    arms["synthetic"] = (syn.astype(np.float32), imgs)

    out = {
        "kind": "rgb_sca_overfit_functionality_check",
        "source": "hardware_traces_plus_synthetic_control",
        "hardware_used": False,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "args": vars(args),
        "note": ("Memorising a few images is a FUNCTIONALITY test. It cannot distinguish "
                 "poor generalization from an uninformative input, and is not evidence "
                 "of reconstruction."),
        "arms": {},
    }

    print(f"budget: {args.epochs} epochs on {args.n} images, lr {args.lr}\n")
    for name in ("synthetic", "control", "natural"):
        x, im = arms[name]
        mu, sd = x.mean(0), x.std(0) + 1e-8
        xs = torch.from_numpy(((x - mu) / sd).astype(np.float32))
        y = torch.from_numpy(im.astype(np.float32) / 255.0)

        torch.manual_seed(args.seed)
        model = G.build_rgb_model(x.shape[1], im.shape[-1], torch, nn)
        opt = torch.optim.Adam(model.parameters(), args.lr)
        mse = nn.MSELoss()
        model.train()
        curve = []
        gnorm = float("nan")
        for ep in range(args.epochs):
            opt.zero_grad()
            loss = G.gradmse(model(xs), y, torch, mse)
            loss.backward()
            gnorm = float(sum((p.grad ** 2).sum() for p in model.parameters()
                              if p.grad is not None) ** 0.5)
            opt.step()
            if ep % max(1, args.epochs // 8) == 0 or ep == args.epochs - 1:
                curve.append({"epoch": ep, "loss": float(loss.detach())})

        model.eval()
        with torch.no_grad():
            rec = model(xs).numpy()
        m = G.rgb_metrics(rec, y.numpy())
        # Prediction spread: a model that collapsed to the dataset mean has ~0 here,
        # which is indistinguishable from "trained" if you only look at the loss.
        spread = float(np.mean(np.std(rec, axis=0)) * 255)
        truth_spread = float(np.mean(np.std(y.numpy(), axis=0)) * 255)
        out["arms"][name] = {
            "train_mae": m["mae_pooled"],
            "train_mssim": float(np.mean(m["mssim_per_channel"])),
            "loss_curve": curve,
            "final_grad_norm": gnorm,
            "prediction_spread_255": spread,
            "truth_spread_255": truth_spread,
            "spread_ratio": spread / max(truth_spread, 1e-9),
        }
        print(f"  {name:10s} train MAE {m['mae_pooled']:6.2f}  MSSIM "
              f"{np.mean(m['mssim_per_channel']):.3f}  loss "
              f"{curve[0]['loss']:.4f} -> {curve[-1]['loss']:.4f}  "
              f"grad {gnorm:.2e}  spread {spread:.2f}/{truth_spread:.2f} "
              f"({spread / max(truth_spread, 1e-9):.2f}x)", flush=True)

    syn_ok = out["arms"]["synthetic"]["train_mae"] < 15.0
    nat_ok = out["arms"]["natural"]["train_mae"] < 15.0
    out["verdict"] = {
        "pipeline_can_fit_a_learnable_mapping": bool(syn_ok),
        "pipeline_can_memorise_natural_traces": bool(nat_ok),
        "reading": (
            "synthetic fits + natural fits -> pipeline is functional; the held-out "
            "failure is generalization or input information, which this test does not "
            "separate. synthetic fits + natural does NOT -> the natural features carry "
            "too little information to memorise even 8 samples. synthetic fails -> "
            "implementation/optimization defect; nothing downstream is interpretable."),
    }
    print(f"\n  synthetic fits: {syn_ok}   natural memorises: {nat_ok}")
    print(f"  {out['verdict']['reading']}")

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "results.json"), "w") as fh:
        json.dump(out, fh, indent=1, default=float)
    print(f"\nwrote {os.path.join(args.out, 'results.json')}")


if __name__ == "__main__":
    main()
