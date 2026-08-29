#!/usr/bin/env python3
import glob
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TRACE_DIR = os.path.join(ROOT, "host", "traces_leakage_60_l63")
RESULT_DIR = os.path.join(ROOT, "attack", "results_leakage_60_l63_p30")
FIG_DIR = os.path.join(ROOT, "report", "paper_figures")
os.makedirs(FIG_DIR, exist_ok=True)
sys.path.insert(0, os.path.join(ROOT, "attack"))

from leakage_first_reconstruct import (  # noqa: E402
    extract_features,
    fit_lookup,
    load_kernel_bits,
    match_counts_for_codes,
    patch_codes_from_patches,
    patch_matrix_from_image,
)


def savefig(name):
    path = os.path.join(FIG_DIR, name)
    plt.savefig(path, dpi=220, bbox_inches="tight")
    plt.close()
    print(path)


def load_manifest():
    with open(os.path.join(TRACE_DIR, "capture_manifest.json")) as fp:
        return json.load(fp)


def validate_dataset(manifest):
    files = sorted(glob.glob(os.path.join(TRACE_DIR, "img*.npz")))
    shapes = []
    for path in files:
        d = np.load(path)
        shapes.append(tuple(d["repeats"].shape))
    report = {
        "n_trace_files": len(files),
        "expected_files": 60,
        "all_repeat_shapes": sorted(set(shapes)),
        "manifest_status": manifest["status"],
        "bitstream_sha256": manifest["bitstream_sha256"],
        "first_trace_file": os.path.basename(files[0]),
        "last_trace_file": os.path.basename(files[-1]),
    }
    with open(os.path.join(ROOT, "report", "hardware_trace_validation.json"), "w") as fp:
        json.dump(report, fp, indent=2)


def raw_trace_figure(manifest):
    d = np.load(os.path.join(TRACE_DIR, "img0030.npz"))
    trace = d["repeats"][0, 0]
    spw = int(manifest["samples_per_window"])
    nwin = 20
    samples = nwin * spw
    t_us = np.arange(samples) / (manifest["fpga_freq_hz"] * manifest["samples_per_cycle"]) * 1e6
    y = trace[:samples] - np.median(trace[:samples])

    plt.figure(figsize=(7.2, 2.6))
    plt.plot(t_us, y, lw=0.8, color="#1f4e79")
    for i in range(nwin + 1):
        plt.axvline(i * spw / (manifest["fpga_freq_hz"] * manifest["samples_per_cycle"]) * 1e6,
                    color="#888888", lw=0.35, alpha=0.45)
    plt.title("Captured CW-Lite power trace, first 20 held windows")
    plt.xlabel("time (us)")
    plt.ylabel("centered ADC value")
    plt.grid(True, alpha=0.25)
    savefig("fig_raw_power_windows.png")


def feature_heatmap_figure(manifest):
    path = os.path.join(TRACE_DIR, "img0030.npz")
    feat = extract_features(path, int(manifest["dwell"]), int(manifest["samples_per_cycle"]),
                            int(manifest["presamples"]), "mean_abs")
    heat = feat[0].reshape(26, 26)
    plt.figure(figsize=(4.3, 3.7))
    im = plt.imshow(heat, cmap="viridis")
    plt.title("Extracted side-channel feature map, kernel 0")
    plt.xlabel("window x")
    plt.ylabel("window y")
    plt.colorbar(im, fraction=0.046, pad=0.04, label="normalized mean abs")
    savefig("fig_feature_heatmap.png")


def lookup_figure(manifest):
    files = sorted(glob.glob(os.path.join(TRACE_DIR, "img*.npz")))
    kernel_bits = load_kernel_bits(manifest["kernels"], int(manifest["n_kernels"]),
                                   manifest["probe_kernels"])
    lookup = fit_lookup(files[:30], kernel_bits, int(manifest["dwell"]),
                        int(manifest["samples_per_cycle"]), int(manifest["presamples"]),
                        "mean_abs")

    plt.figure(figsize=(5.2, 3.2))
    for kid in range(3):
        plt.plot(range(10), lookup[kid], marker="o", lw=1.3, label=f"probe {kid}")
    plt.title("Profiled relation between match count and leakage")
    plt.xlabel("3x3 XNOR match count")
    plt.ylabel("normalized feature")
    plt.grid(True, alpha=0.25)
    plt.legend(frameon=False, fontsize=8)
    savefig("fig_profile_lookup.png")


def reconstruction_figure():
    files = sorted(glob.glob(os.path.join(RESULT_DIR, "recovered_*.npz")))[:6]
    fig, axes = plt.subplots(2, len(files), figsize=(7.2, 2.7))
    for i, path in enumerate(files):
        d = np.load(path)
        axes[0, i].imshow(d["original"], cmap="gray", vmin=0, vmax=1)
        axes[1, i].imshow(d["recovered"], cmap="gray", vmin=0, vmax=1)
        axes[0, i].set_title(f"label {int(d['label'])}", fontsize=8)
        axes[0, i].axis("off")
        axes[1, i].axis("off")
    axes[0, 0].set_ylabel("input", fontsize=8)
    axes[1, 0].set_ylabel("recovered", fontsize=8)
    plt.tight_layout(pad=0.25)
    savefig("fig_reconstruction_examples.png")


def score_hist_figure():
    with open(os.path.join(RESULT_DIR, "summary.json")) as fp:
        data = json.load(fp)
    vals = [row["bit_acc"] for row in data["images"]]
    plt.figure(figsize=(5.0, 3.0))
    plt.hist(vals, bins=np.linspace(0.94, 1.0, 13), color="#4c78a8", edgecolor="white")
    plt.axvline(data["summary"]["bit_acc"], color="#d62728", lw=1.5,
                label=f"mean {data['summary']['bit_acc']:.3f}")
    plt.axvline(data["summary"]["all_zero_bit_acc"], color="#666666", lw=1.2,
                linestyle="--", label="all-zero baseline")
    plt.title("Held-out image reconstruction accuracy")
    plt.xlabel("bit accuracy")
    plt.ylabel("number of images")
    plt.legend(frameon=False, fontsize=8)
    savefig("fig_accuracy_hist.png")


def patch_count_figure(manifest):
    d = np.load(os.path.join(RESULT_DIR, "recovered_0030.npz"))
    kernel_bits = load_kernel_bits(manifest["kernels"], int(manifest["n_kernels"]),
                                   manifest["probe_kernels"])
    code_counts = match_counts_for_codes(kernel_bits)
    pred_counts = code_counts[:, d["pred_codes"]]
    true_counts = code_counts[:, d["true_codes"]]
    plt.figure(figsize=(7.2, 2.6))
    for kid in range(3):
        plt.plot(pred_counts[kid, :80], lw=1.0, label=f"pred probe {kid}")
        plt.plot(true_counts[kid, :80], lw=0.8, color="black", alpha=0.25)
    plt.title("Decoded match-count stream after template matching")
    plt.xlabel("window index")
    plt.ylabel("match count")
    plt.grid(True, alpha=0.25)
    plt.legend(frameon=False, fontsize=8)
    savefig("fig_match_count_stream.png")


def main():
    manifest = load_manifest()
    validate_dataset(manifest)
    raw_trace_figure(manifest)
    feature_heatmap_figure(manifest)
    lookup_figure(manifest)
    patch_count_figure(manifest)
    reconstruction_figure()
    score_hist_figure()


if __name__ == "__main__":
    main()
