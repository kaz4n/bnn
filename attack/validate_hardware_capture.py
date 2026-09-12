#!/usr/bin/env python3
"""Validate that a trace directory is suitable to call a hardware capture.

This is not a cryptographic proof. It is an evidence gate for the lab workflow:
the manifest must look like a live CW305/ChipWhisperer run, must not be marked
as simulation, and the stored trace arrays must contain finite non-trivial
signals. New capture scripts also store scope/target serials and a bit-exact
functional check, and this validator reports them.
"""
import argparse
import glob
import json
import os
import time

import numpy as np


def load_json(path):
    with open(path, "r", encoding="utf-8") as fp:
        return json.load(fp)


def trace_stats(arr):
    x = np.asarray(arr, dtype=np.float64)
    return {
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "finite": bool(np.all(np.isfinite(x))),
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
    }


def inspect_img_files(files):
    samples = []
    for path in files:
        d = np.load(path)
        key = "repeats" if "repeats" in d.files else "traces"
        if key not in d.files:
            raise ValueError(f"{path} has no traces/repeats array")
        samples.append({
            "file": os.path.basename(path),
            "array": key,
            "stats": trace_stats(d[key]),
            "has_image": "image" in d.files,
            "has_label": "label" in d.files,
            "trace_source": str(d["trace_source"]) if "trace_source" in d.files else None,
        })
    return samples


def inspect_shards(files):
    samples = []
    for path in files:
        d = np.load(path)
        if "traces" not in d.files:
            raise ValueError(f"{path} has no traces array")
        samples.append({
            "file": os.path.basename(path),
            "array": "traces",
            "stats": trace_stats(d["traces"]),
            "has_images": "images" in d.files,
            "has_labels": "labels" in d.files,
            "n_items": int(len(d["traces"])),
        })
    return samples


def validate_manifest(manifest):
    issues = []
    warnings = []
    source = str(manifest.get("trace_source", "")).lower()
    mode = str(manifest.get("capture_mode", "")).lower()
    if manifest.get("status") != "complete":
        issues.append(f"manifest status is {manifest.get('status')!r}, not 'complete'")
    if "sim" in source or "sim" in mode:
        issues.append("manifest source/mode looks simulated")
    if not any(token in source for token in ("cw", "analog", "ro", "tdc")):
        issues.append(f"trace_source does not look like a supported hardware channel: {source!r}")
    if manifest.get("hardware_run") is False:
        issues.append("hardware_run is explicitly false")
    if not manifest.get("bitstream_sha256"):
        warnings.append("missing bitstream_sha256")
    if not manifest.get("scope_serial"):
        warnings.append("missing scope_serial; older captures may not have this")
    if not manifest.get("target_serial"):
        warnings.append("missing target_serial; older captures may not have this")
    if manifest.get("adc_locked_initial") is False:
        issues.append("ADC was not locked at capture start")
    if manifest.get("adc_locked_final") is False:
        issues.append("ADC was not locked at capture end")
    fc = manifest.get("functional_check")
    if isinstance(fc, dict):
        if fc.get("passed") is False:
            issues.append("functional_check reports failure")
        if fc.get("skipped") is True:
            warnings.append("functional_check was skipped")
    else:
        warnings.append("missing functional_check object")
    return issues, warnings


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", required=True, help="capture directory")
    ap.add_argument("--out", default=None, help="optional JSON report")
    ap.add_argument("--max-files", type=int, default=5)
    ap.add_argument("--min-std", type=float, default=1e-7)
    args = ap.parse_args()

    manifest_path = os.path.join(args.traces, "capture_manifest.json")
    manifest = load_json(manifest_path)
    issues, warnings = validate_manifest(manifest)

    img_files = sorted(glob.glob(os.path.join(args.traces, "img*.npz")))[:args.max_files]
    shard_files = sorted(glob.glob(os.path.join(args.traces, "shard*.npz")))[:args.max_files]
    if img_files:
        samples = inspect_img_files(img_files)
        capture_kind = "img_files"
    elif shard_files:
        samples = inspect_shards(shard_files)
        capture_kind = "shards"
    else:
        samples = []
        capture_kind = "empty"
        issues.append("no img*.npz or shard*.npz files found")

    for sample in samples:
        st = sample["stats"]
        if not st["finite"]:
            issues.append(f"{sample['file']} contains non-finite trace values")
        if st["std"] <= args.min_std:
            issues.append(f"{sample['file']} trace std {st['std']:.3g} is too small")
        if st["max"] == st["min"]:
            issues.append(f"{sample['file']} trace is constant")

    report = {
        "kind": "cw305_hardware_capture_validation",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "trace_dir": os.path.abspath(args.traces),
        "capture_kind": capture_kind,
        "passed": not issues,
        "issues": issues,
        "warnings": warnings,
        "manifest_summary": {
            key: manifest.get(key) for key in [
                "status", "trace_source", "capture_mode", "design",
                "chipwhisperer_version", "scope_serial", "target_serial",
                "bitstream_sha256", "adc_locked_initial", "adc_locked_final",
                "n_images", "n_kernels", "avg", "shuffle_seed",
            ]
        },
        "functional_check": manifest.get("functional_check"),
        "sample_files": samples,
    }
    text = json.dumps(report, indent=2)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fp:
            fp.write(text)
            fp.write("\n")
    print(text)
    if issues:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
