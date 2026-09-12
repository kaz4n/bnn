#!/usr/bin/env python3
"""Phase 0 driver: architectural feasibility of RGB input recovery. SIMULATION ONLY.

Runs both probes across the three channel dataflows and writes a result artifact.
No hardware is touched. No bitstream is built or programmed. No ChipWhisperer capture
is performed. Every record written by this script is labelled source="simulation" so it
can never be mistaken for a bench measurement -- the failure mode the project's own
audit (report/RGB_FEASIBILITY_AND_EVIDENCE_AUDIT_2026-09-11.md, Finding 1) found in
older result files.

What a positive result here does and does not mean
--------------------------------------------------
DOES  : per-channel information survives the first-layer accumulation for this
        dataflow and kernel set, and is accessible to a linear read-out at zero noise.
        That is the architectural ceiling.
DOES NOT : say anything about recoverability on the CW305/CW-Lite bench, which has a
        separate and independently limiting sampling constraint (4 samples/cycle vs the
        2.5 GS/s used by Wei et al.). Bench modelling is Phase 1 and is deliberately
        absent here so the two limits cannot be confused.

Usage:
    python -m experiments.rgb_sca.run_phase0 --size 32 --out results/phase0_<date>
    python -m experiments.rgb_sca.run_phase0 --images-npz my_scenes.npz   # (N,C,H,W) uint8
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import platform
import sys
import time

import numpy as np

from . import rgb_forward as F
from . import rgb_probe as P

ASSETS = os.path.join(os.path.dirname(__file__), "..", "rgb_instrumented", "assets")
RESULT_KIND = "rgb_sca_phase0_simulation"


def load_images(size: int, npz: str | None, n_aug: int, seed: int):
    """Load RGB scenes as (N, C, H, W) uint8, plus provenance describing independence.

    The augmented copies are rolls of the source photos. They preserve natural-image
    statistics but are NOT independent scenes, and the returned provenance says so
    explicitly so no downstream report can quietly count them as such.
    """
    if npz:
        arr = np.load(npz)["images"].astype(np.uint8)
        if arr.ndim != 4 or arr.shape[1] != 3:
            raise SystemExit(f"--images-npz must hold (N,3,H,W) uint8; got {arr.shape}")
        return arr, {"source_file": npz, "n_independent_scenes": int(len(arr)),
                     "augmented": 0}

    from PIL import Image
    paths = sorted(p for p in glob.glob(os.path.join(ASSETS, "*"))
                   if p.lower().endswith((".png", ".jpg", ".jpeg")))
    if not paths:
        raise SystemExit(f"no images found in {ASSETS}; pass --images-npz")
    base = []
    for p in paths:
        im = Image.open(p).convert("RGB").resize((size, size), Image.LANCZOS)
        base.append(np.asarray(im, dtype=np.uint8).transpose(2, 0, 1))
    base = np.stack(base)

    rng = np.random.default_rng(seed)
    extra = []
    for i in range(n_aug):
        src = base[i % len(base)]
        extra.append(np.roll(src, (int(rng.integers(0, size)), int(rng.integers(0, size))),
                             axis=(1, 2)))
    images = np.concatenate([base, np.stack(extra)]) if extra else base
    # Carry the ORIGINATING scene id through augmentation. Splitting by array position
    # put rolled copies of a training scene into the held-out set -- with the documented
    # 4+16 construction and seed 0, every held-out row came from a scene already seen in
    # training, so the pilot could not speak to unseen-scene generalization at all.
    # Found in external review, 12 September 2026.
    scene_id = np.concatenate([np.arange(len(base)),
                               np.array([i % len(base) for i in range(len(extra))],
                                        dtype=int)]) if extra else np.arange(len(base))
    return images, {
        "source_files": [os.path.basename(p) for p in paths],
        "n_independent_scenes": int(len(base)),
        "augmented": int(len(extra)),
        "augmentation": "cyclic roll of a source photo; NOT an independent scene",
        "scene_id": scene_id.tolist(),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--size", type=int, default=32, choices=[32, 64, 128])
    ap.add_argument("--kernels", type=int, default=9,
                    help="probe kernel count; Wei et al. S7.3 used 9")
    ap.add_argument("--ksize", type=int, default=3, choices=[3, 5])
    ap.add_argument("--images-npz", default=None, help="(N,3,H,W) uint8 scenes")
    ap.add_argument("--augment", type=int, default=16,
                    help="rolled copies added to the source photos (not independent)")
    ap.add_argument("--noise", type=float, nargs="*", default=[0.0, 0.1, 0.5, 1.0],
                    help="crude Gaussian noise sweep, in units of per-image power std")
    ap.add_argument("--context", type=int, default=1)
    ap.add_argument("--alpha", type=float, default=1e3, help="ridge regularization")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="directory for results.json")
    args = ap.parse_args()

    t0 = time.time()
    images, prov = load_images(args.size, args.images_npz, args.augment, args.seed)
    kernels = F.random_kernels(args.kernels, C=3, K=args.ksize, seed=args.seed + 1)
    sym = np.repeat(F.random_kernels(args.kernels, C=1, K=args.ksize, seed=args.seed + 2),
                    3, axis=1)

    print(f"images {images.shape}  ({prov['n_independent_scenes']} independent scenes"
          f" + {prov.get('augmented', 0)} augmented)")
    print(f"kernels {kernels.shape}  channel-asymmetry "
          f"{P.kernel_channel_asymmetry(kernels):.2f}\n")

    out = {
        "kind": RESULT_KIND,
        "source": "simulation",
        "hardware_used": False,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "args": vars(args),
        "provenance": prov,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "kernel_channel_asymmetry": P.kernel_channel_asymmetry(kernels),
        "distinguisher": {},
        "distinguisher_symmetric_control": {},
        "recovery": {},
    }

    print("PROBE 1 - channel-permutation distinguisher (exact, noise-free)")
    for df in F.DATAFLOWS:
        r = P.distinguisher(images[0], kernels, df)
        c = P.distinguisher(images[0], sym, df)
        out["distinguisher"][df] = r
        out["distinguisher_symmetric_control"][df] = c
        print(f"  {df:9s} identical={str(r['identical']):5s} "
              f"frac_diff={r['frac_differing']:.3f} rel_L1={r['rel_l1']:.4f}"
              f"   | symmetric-kernel control identical={str(c['identical'])}")

    print("\nPROBE 2 - linear recovery vs controls, MAE in 0-255 units")
    for df in F.DATAFLOWS:
        out["recovery"][df] = {}
        print(f"\n  {df}")
        for nz in args.noise:
            r = P.recovery_probe(images, kernels, df, alpha=args.alpha,
                                 context=args.context, noise=nz, seed=args.seed,
                                 scene_id=prov.get("scene_id"))
            out["recovery"][df][f"noise_{nz}"] = r
            t = r["trace"]["mae_per_channel"]
            print(f"    noise={nz:<4} trace R{t[0]:6.2f} G{t[1]:6.2f} B{t[2]:6.2f}"
                  f" | prior {r['prior_only']['mae_pooled']:6.2f}"
                  f" | shuffled {r['shuffled']['mae_pooled']:6.2f}"
                  f" | advantage {r['advantage_over_prior_pooled']:+6.2f}")

    out["elapsed_sec"] = round(time.time() - t0, 2)

    if args.out:
        os.makedirs(args.out, exist_ok=True)
        path = os.path.join(args.out, "results.json")
        with open(path, "w") as fh:
            json.dump(out, fh, indent=1)
        print(f"\nwrote {path}")
    return out


if __name__ == "__main__":
    main()
