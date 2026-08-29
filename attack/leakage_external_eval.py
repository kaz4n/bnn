#!/usr/bin/env python3
"""Decode an external eval capture using profile traces from another directory."""
import argparse
import glob
import json
import os

import numpy as np

from leakage_first_reconstruct import (
    N_WINDOWS,
    decode_patches,
    extract_features,
    fit_lookup,
    load_kernel_bits,
    match_counts_for_codes,
    metrics,
    patch_codes_from_patches,
    patch_matrix_from_image,
    reconstruct_from_codes,
    summarize,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile-traces", required=True)
    ap.add_argument("--eval-traces", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--profile-count", type=int, default=30)
    ap.add_argument("--feature", choices=["mean_abs", "rms", "p2p"], default="mean_abs")
    args = ap.parse_args()

    with open(os.path.join(args.profile_traces, "capture_manifest.json")) as fp:
        profile_manifest = json.load(fp)
    with open(os.path.join(args.eval_traces, "capture_manifest.json")) as fp:
        eval_manifest = json.load(fp)

    n_kernels = int(profile_manifest["n_kernels"])
    dwell = int(profile_manifest["dwell"])
    spc = int(profile_manifest["samples_per_cycle"])
    presamples = int(profile_manifest.get("presamples", 0))
    kernel_bits = load_kernel_bits(profile_manifest["kernels"], n_kernels,
                                   profile_manifest.get("probe_kernels", "onehot"))
    code_counts = match_counts_for_codes(kernel_bits)

    profile_files = sorted(glob.glob(os.path.join(args.profile_traces, "img*.npz")))
    eval_files = sorted(glob.glob(os.path.join(args.eval_traces, "img*.npz")))
    if len(profile_files) < args.profile_count:
        raise ValueError("not enough profile files")
    if not eval_files:
        raise ValueError("no eval files")

    lookup = fit_lookup(profile_files[:args.profile_count], kernel_bits, dwell, spc,
                        presamples, args.feature)
    os.makedirs(args.out, exist_ok=True)
    rows = []
    for path in eval_files:
        d = np.load(path)
        img = d["image"].astype(np.uint8)
        true_codes = patch_codes_from_patches(patch_matrix_from_image(img))
        feat = extract_features(path, dwell, spc, presamples, args.feature)
        pred_codes = decode_patches(feat, lookup, code_counts)
        recon = reconstruct_from_codes(pred_codes)
        row = metrics(recon, img)
        row["patch_exact"] = float(np.mean(pred_codes == true_codes))
        row["label"] = int(d["label"])
        row["file"] = os.path.basename(path)
        row["trace_std_mean"] = float(np.mean(np.std(d["repeats"], axis=2)))
        row["trace_abs_mean"] = float(np.mean(np.abs(d["repeats"])))
        rows.append(row)
        outname = f"recovered_{os.path.basename(path)[3:7]}.npz"
        np.savez(os.path.join(args.out, outname), original=img, recovered=recon,
                 pred_codes=pred_codes, true_codes=true_codes, label=int(d["label"]))

    summary = summarize(rows)
    payload = {
        "summary": summary,
        "images": rows,
        "profile_manifest": profile_manifest,
        "eval_manifest": eval_manifest,
        "feature": args.feature,
        "n_windows": N_WINDOWS,
    }
    with open(os.path.join(args.out, "summary.json"), "w") as fp:
        json.dump(payload, fp, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
