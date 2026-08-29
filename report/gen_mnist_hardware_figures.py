from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "attack"))

import power_extract  # noqa: E402


FIG_DIR = ROOT / "report" / "figs"
TRACE_DIR = ROOT / "host" / "traces_analog_trigfix_500"
NOFIT_DIR = ROOT / "attack" / "results_trigfix500_nofit_norm"
FIT_DIR = ROOT / "attack" / "results_trigfix500_curvefit_norm"


def _norm(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    lo, hi = np.percentile(x, [1, 99])
    if hi <= lo:
        lo, hi = float(x.min()), float(x.max())
    return np.clip((x - lo) / max(hi - lo, 1e-12), 0.0, 1.0)


def _load_summary(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)["summary"]


def plot_trace_transformations() -> None:
    z = np.load(TRACE_DIR / "img0300.npz")
    raw = z["traces"][0].astype(np.float64)
    spc = int(z["samples_per_cycle"])
    n_cycles = int(z["n_cycles"])
    sample_rate = float(z["sample_rate_hz"])
    ew = int(z["ew"])
    cycle0_offset = int(z["cycle0_offset"])

    dc = power_extract.dc_restore_highpass_inverse(raw, sample_rate, 250.0)
    lp = power_extract.low_pass_filter(dc, sample_rate, 60e6)
    windows = power_extract.align_cycle_windows(lp, spc, n_cycles)

    # Keep the RC-fit panel bounded; the full 900-cycle fit is slow and not visually
    # different for this diagnostic plot.
    vis_cycles = 260
    nofit_pc = power_extract.extract_per_cycle_paper(
        z["traces"][[0]],
        spc,
        vis_cycles,
        sample_rate,
        60e6,
        do_curve_fit=False,
    )[0]
    fit_pc = power_extract.extract_per_cycle_paper(
        z["traces"][[0]],
        spc,
        vis_cycles,
        sample_rate,
        60e6,
        do_curve_fit=True,
    )[0]
    full_fit_pc = power_extract.extract_per_cycle_paper(
        z["traces"][[0]],
        spc,
        n_cycles,
        sample_rate,
        60e6,
        do_curve_fit=False,
    )
    heat = power_extract.remap_extended_to_output(
        full_fit_pc, ew, 3, cycle0_offset=cycle0_offset
    )[0].reshape(28, 28)

    xs = np.arange(2600) / sample_rate * 1e6
    fig, axes = plt.subplots(3, 2, figsize=(12, 9), constrained_layout=True)
    fig.suptitle("MNIST Hardware Trace Transformation Pipeline", fontsize=16)

    axes[0, 0].plot(xs, raw[:2600], lw=0.9, color="#335c81")
    axes[0, 0].set_title("Raw CW-Lite analog trace")
    axes[0, 0].set_xlabel("Time (us)")
    axes[0, 0].set_ylabel("ADC units")

    axes[0, 1].plot(xs, dc[:2600], lw=0.9, color="#6a4c93")
    axes[0, 1].set_title("After DC restoration")
    axes[0, 1].set_xlabel("Time (us)")
    axes[0, 1].set_ylabel("Restored amplitude")

    axes[1, 0].plot(xs, lp[:2600], lw=0.9, color="#2a9d8f")
    for st, _ in windows[:12]:
        axes[1, 0].axvline(st / sample_rate * 1e6, color="#111111", alpha=0.15, lw=0.8)
    axes[1, 0].set_title("Low-pass + cycle alignment marks")
    axes[1, 0].set_xlabel("Time (us)")
    axes[1, 0].set_ylabel("Filtered amplitude")

    cyc = np.arange(vis_cycles)
    axes[1, 1].plot(cyc, _norm(nofit_pc), label="No curve fit", lw=1.2, color="#e76f51")
    axes[1, 1].plot(cyc, _norm(fit_pc), label="Curve fit", lw=1.2, color="#264653")
    axes[1, 1].set_title("Per-cycle feature after extraction")
    axes[1, 1].set_xlabel("Cycle")
    axes[1, 1].set_ylabel("Normalized energy")
    axes[1, 1].legend(frameon=False)

    im = axes[2, 0].imshow(_norm(heat), cmap="magma", interpolation="nearest")
    axes[2, 0].set_title("Mapped 28x28 power image, kernel 0")
    axes[2, 0].set_xticks([])
    axes[2, 0].set_yticks([])
    fig.colorbar(im, ax=axes[2, 0], fraction=0.046, pad=0.04)

    axes[2, 1].axis("off")
    metadata = (
        "Capture metadata\n"
        f"trigger: {str(z['trigger_source'])}\n"
        f"sample rate: {sample_rate/1e6:.1f} MS/s\n"
        f"FPGA clock: {float(z['fpga_freq_hz'])/1e6:.1f} MHz\n"
        f"samples/cycle: {spc}\n"
        f"averaging: {int(z['avg'])}x\n"
        f"cycle0 offset: {cycle0_offset} cycles\n"
        f"trace length: {raw.size} samples"
    )
    axes[2, 1].text(0.02, 0.96, metadata, va="top", ha="left", family="monospace", fontsize=11)

    fig.savefig(FIG_DIR / "mnist_trace_transformations.png", dpi=180)
    plt.close(fig)


def plot_recovery_examples() -> None:
    files = ["img0300.npz", "img0308.npz", "img0340.npz", "img0401.npz"]
    fig, axes = plt.subplots(len(files), 4, figsize=(8.5, 8.8), constrained_layout=True)
    fig.suptitle("Recovered MNIST Images from Corrected Hardware Captures", fontsize=15)

    col_titles = ["Original", "Background\n(curve fit)", "Template\n(curve fit)", "Background\n(no fit)"]
    for ax, title in zip(axes[0], col_titles):
        ax.set_title(title)

    for row, name in enumerate(files):
        fit = np.load(FIT_DIR / name)
        nofit = np.load(NOFIT_DIR / name)
        imgs = [
            fit["original"],
            fit["background"],
            fit["template"],
            nofit["background"],
        ]
        for col, img in enumerate(imgs):
            if col == 2:
                hi = float(np.max(img)) if float(np.max(img)) > 0 else 1.0
                axes[row, col].imshow(img, cmap="gray", vmin=0, vmax=hi, interpolation="nearest")
            else:
                axes[row, col].imshow(img, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
        axes[row, 0].set_ylabel(f"{name[3:7]}\nlabel {int(fit['label'])}", rotation=0, labelpad=24, va="center")

    fig.savefig(FIG_DIR / "mnist_recovery_examples.png", dpi=180)
    plt.close(fig)


def plot_metric_summary() -> None:
    nofit = _load_summary(NOFIT_DIR / "summary.json")
    fit = _load_summary(FIT_DIR / "summary.json")

    labels = ["No fit", "Curve fit", "Paper target"]
    bg_rec = [nofit["recog_acc_background"], fit["recog_acc_background"], fit["paper_targets"]["recog_background"]]
    tm_rec = [nofit["recog_acc_template"], fit["recog_acc_template"], fit["paper_targets"]["recog_template"]]
    bg_pix = [nofit["bg_pixel_acc_mean"], fit["bg_pixel_acc_mean"], np.nan]

    x = np.arange(len(labels))
    width = 0.25
    fig, ax = plt.subplots(figsize=(9.0, 5.2), constrained_layout=True)
    ax.bar(x - width, bg_rec, width, label="Background recognition", color="#335c81")
    ax.bar(x, tm_rec, width, label="Template recognition", color="#e76f51")
    ax.bar(x + width, bg_pix, width, label="Background pixel accuracy", color="#2a9d8f")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Accuracy / score")
    ax.set_xticks(x, labels)
    ax.set_title("MNIST Recovery Performance on Corrected Hardware Captures")
    ax.legend(frameon=False, loc="upper left")
    ax.grid(axis="y", alpha=0.2)
    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", fontsize=8, padding=2)
    fig.savefig(FIG_DIR / "mnist_metric_summary.png", dpi=180)
    plt.close(fig)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plot_trace_transformations()
    plot_recovery_examples()
    plot_metric_summary()
    print(f"Wrote figures to {FIG_DIR}")


if __name__ == "__main__":
    main()
