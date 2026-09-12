#!/usr/bin/env python3
"""Concatenate several power templates into one multi-condition template.

Item 3 of the next-steps list: a template built at one clock dies when the clock
changes (F1 0.843 -> 0.007 at 10 MHz). The question is whether profiling under
several conditions at once buys tolerance.

The template is just a table of (power vector, patch label) rows, so mixing
conditions is a concatenation. The only thing that needs care is the row count:
a bigger template at a fixed search radius returns more candidates per window and
scores worse, which is exactly what the 300-image run showed. So mix templates
that were each built from FEWER images, keeping the total profiling image count
equal to the single-condition baseline.
"""
import argparse
import json
import os

import numpy as np

ARRAYS = ("rho", "patch_code", "patch_bits", "source_file", "source_window")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True,
                    help="template .npz files to concatenate")
    ap.add_argument("--image-slices", nargs="*", default=None,
                    help="one A:B per input; keep only rows whose source image is in "
                         "that positional slice of the input's sorted unique images. "
                         "Use it to give each condition a DIFFERENT set of images, so "
                         "the merged template has the same image count as the "
                         "single-condition baseline rather than more.")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    if args.image_slices and len(args.image_slices) != len(args.inputs):
        raise SystemExit("--image-slices needs one A:B per input")

    parts = [np.load(p, allow_pickle=False) for p in args.inputs]
    kept = []
    for i, part in enumerate(parts):
        sel = {k: part[k] for k in ARRAYS}
        if args.image_slices:
            lo, hi = (int(v) for v in args.image_slices[i].split(":"))
            images = sorted(set(part["source_file"].tolist()))
            chosen = set(images[lo:hi])
            mask = np.isin(part["source_file"], list(chosen))
            sel = {k: v[mask] for k, v in sel.items()}
            print(f"{os.path.basename(args.inputs[i])}: images[{lo}:{hi}] -> "
                  f"{len(chosen)} images, {int(mask.sum())} rows")
        kept.append(sel)

    merged = {}
    for key in ARRAYS:
        merged[key] = np.concatenate([k[key] for k in kept], axis=0)

    manifests = [json.loads(str(p["manifest_json"])) for p in parts]
    total_images = len(set(merged["source_file"].tolist()))
    combined = {
        "kind": "cw305_active_power_template_merged",
        "n_sources": len(parts),
        "entries": int(merged["rho"].shape[0]),
        "profile_count": total_images,
        "sources": [
            {"file": os.path.abspath(p), "manifest": m}
            for p, m in zip(args.inputs, manifests)
        ],
        "note": ("rows from several capture conditions in one table; "
                 "the search radius is unchanged, so keep the total profiling "
                 "image count equal to the single-condition baseline"),
    }
    merged["manifest_json"] = np.asarray(json.dumps(combined))

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    np.savez_compressed(args.out, **merged)
    print(json.dumps({"out": os.path.abspath(args.out),
                      "entries": combined["entries"],
                      "profile_count": total_images,
                      "sources": len(parts)}, indent=2))


if __name__ == "__main__":
    main()
