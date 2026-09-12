#!/usr/bin/env python3
"""Synthesise the trace set a clock-randomising defender would produce.

Gap 1 shows a power template is bound to the clock it was profiled at. That makes
clock randomisation a candidate countermeasure: if the accelerator picks a
different clock for every inference, the attacker's fixed-clock template should
be wrong most of the time.

Because the 4-frequency evaluation grid captured every image at every clock, the
defended trace set can be assembled rather than recaptured: for each image, keep
the trace from one randomly chosen clock. That is exactly what a defender varying
the clock per inference would hand the attacker, and it costs no board time.

The manifest records that the set is synthetic and which clock each image came
from, so nothing downstream can mistake it for a single-condition capture.
"""
import argparse
import glob
import json
import os
import shutil

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="+", required=True,
                    help="per-frequency capture directories of the SAME images")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    per_dir = {}
    for d in args.dirs:
        files = sorted(os.path.basename(p) for p in glob.glob(os.path.join(d, "img*.npz")))
        if not files:
            raise SystemExit(f"no img*.npz in {d}")
        per_dir[d] = set(files)

    common = sorted(set.intersection(*per_dir.values()))
    if not common:
        raise SystemExit("the directories share no image files")

    rng = np.random.default_rng(args.seed)
    os.makedirs(args.out, exist_ok=True)
    choice = {}
    for name in common:
        src = args.dirs[int(rng.integers(len(args.dirs)))]
        shutil.copyfile(os.path.join(src, name), os.path.join(args.out, name))
        choice[name] = os.path.basename(src)

    base = json.load(open(os.path.join(args.dirs[0], "capture_manifest.json"),
                          encoding="utf-8"))
    base.update({
        "kind": "cw305_leakage_capture_synthetic_jitter",
        "synthetic": True,
        "synthetic_note": ("each image keeps the trace from one randomly chosen clock, "
                           "modelling a defender that varies the accelerator clock per "
                           "inference; assembled from the 4-frequency grid, not recaptured"),
        "source_dirs": [os.path.abspath(d) for d in args.dirs],
        "clock_per_image": choice,
        "fpga_freq_hz": None,
        "seed": args.seed,
    })
    with open(os.path.join(args.out, "capture_manifest.json"), "w", encoding="utf-8") as fp:
        json.dump(base, fp, indent=2)

    counts = {}
    for v in choice.values():
        counts[v] = counts.get(v, 0) + 1
    print(json.dumps({"out": os.path.abspath(args.out), "images": len(common),
                      "clock_mix": counts}, indent=2))


if __name__ == "__main__":
    main()
