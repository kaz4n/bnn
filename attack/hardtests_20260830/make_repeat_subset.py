#!/usr/bin/env python3
"""Create a trace directory holding only the first R hardware repeats per capture.

The averaging sweep must not be confounded. Rather than recapturing at several
--avg settings (different images, different thermal state, different time), we
capture ONCE at avg=5 and subset the un-averaged `repeats` axis afterwards. Every
point of the sweep therefore sees the same images, the same board temperature and
the same gain; only the number of averaged hardware repeats changes.
"""
import argparse
import glob
import json
import os
import shutil

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--repeats", type=int, required=True)
    args = ap.parse_args()

    os.makedirs(args.dst, exist_ok=True)
    files = sorted(glob.glob(os.path.join(args.src, "img*.npz")))
    if not files:
        raise SystemExit(f"no img*.npz in {args.src}")

    for src in files:
        d = np.load(src)
        if "repeats" not in d.files:
            raise SystemExit(f"{src} has no un-averaged repeats axis")
        reps = d["repeats"]
        if args.repeats > reps.shape[1]:
            raise SystemExit(
                f"asked for {args.repeats} repeats but capture only has {reps.shape[1]}")
        sub = reps[:, :args.repeats, :]
        out = {k: d[k] for k in d.files}
        out["repeats"] = sub
        # keep `traces` consistent with the subset actually used
        out["traces"] = sub.mean(axis=1).astype(np.float32)
        np.savez_compressed(os.path.join(args.dst, os.path.basename(src)), **out)

    manifest_path = os.path.join(args.src, "capture_manifest.json")
    with open(manifest_path, "r", encoding="utf-8") as fp:
        manifest = json.load(fp)
    manifest["avg"] = int(args.repeats)
    manifest["derived_from"] = os.path.abspath(args.src)
    manifest["derivation"] = (
        f"first {args.repeats} of {manifest.get('avg')} captured hardware repeats; "
        "no recapture, identical images/thermal state/gain")
    with open(os.path.join(args.dst, "capture_manifest.json"), "w", encoding="utf-8") as fp:
        json.dump(manifest, fp, indent=2)

    print(json.dumps({"dst": os.path.abspath(args.dst),
                      "files": len(files),
                      "repeats": args.repeats}, indent=2))


if __name__ == "__main__":
    main()
