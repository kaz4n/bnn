#!/usr/bin/env python3
"""Figures for the 2026-08-30 catch-up report.

Every figure that shows a power signal is drawn from a real .npz written by the
ChipWhisperer capture scripts. Nothing here is simulated. Figures that are pure
explanation (signal chain, algorithm flow, geometry) are drawn with matplotlib
primitives and are clearly labelled as diagrams.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
HOST = os.path.join(ROOT, "host")
HT = os.path.join(ROOT, "attack", "hardtests_20260830")
RERUN = os.path.join(ROOT, "attack", "rerun_20260830")
FIG = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(FIG, exist_ok=True)

OUT_SIDE = 26
N_WINDOWS = OUT_SIDE * OUT_SIDE

INK = "#12263a"
# One blue-slate ramp instead of red/green/amber. Status is shown by fill
# darkness, border weight and the wording, never by hue, so the figures stay
# readable in greyscale and carry no traffic-light connotation.
DEEP = "#1f3a5f"     # strongest emphasis
ACC2 = "#3d6ea8"     # primary accent
STEEL = "#6b8bb0"    # secondary
MUTE = "#8a97a6"     # de-emphasised
ACC = DEEP           # legacy name: "problem" boxes, now deep slate
GOOD = ACC2          # legacy name: "resolved" boxes, now primary accent
WARN = STEEL         # legacy name: "open" boxes, now secondary
FILL_STRONG = "#dde5ef"
FILL_MID = "#e9eef5"
FILL_SOFT = "#f4f6fa"

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "axes.edgecolor": INK,
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.6,
    "figure.facecolor": "white",
    "savefig.bbox": "tight",
})


def save(fig, name):
    path = os.path.join(FIG, name)
    fig.savefig(path, facecolor="white")
    plt.close(fig)
    print("wrote", os.path.relpath(path, ROOT))


def load_json(path):
    with open(path, "r", encoding="utf-8") as fp:
        return json.load(fp)


def window_features(trace_1d, spw=32, n_windows=N_WINDOWS, presamples=0):
    """Mean absolute power per convolution window -- the feature the attack uses."""
    seg = trace_1d[presamples:presamples + n_windows * spw]
    return np.abs(seg.reshape(n_windows, spw)).mean(axis=1)


def box(ax, x, y, w, h, text, fc="#eef3f8", ec=INK, fs=9, weight="normal", tc=None):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.02",
                                linewidth=1.3, facecolor=fc, edgecolor=ec))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            weight=weight, color=tc or INK, linespacing=1.45)


def arrow(ax, x1, y1, x2, y2, color=INK, style="-|>", lw=1.5, ls="-"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                                 mutation_scale=13, linewidth=lw, color=color,
                                 linestyle=ls, shrinkA=1, shrinkB=1))


def blank(ax):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.grid(False)


# ---------------------------------------------------------------- fig 1
def fig_signal_chain():
    fig, ax = plt.subplots(figsize=(11.6, 4.5))
    blank(ax)
    ax.set_title("Measurement chain. The BNN layer runs on the FPGA; "
                 "only the host control and the scoring run on the PC.", loc="left",
                 fontsize=11.5, weight="bold", pad=12)

    box(ax, 0.015, 0.55, 0.20, 0.30,
        "Host PC\n(Python + ChipWhisperer)\nwrites image + kernel,\nasserts GO", fc="#f2f5f8")
    box(ax, 0.255, 0.55, 0.195, 0.30,
        "CW-Lite capture board\nclock generator +\n10-bit ADC, phase locked", fc=FILL_MID)
    box(ax, 0.585, 0.55, 0.205, 0.30,
        "CW305 Artix-7\nXC7A100T\ncw305_leakage_top\n3x3 binary conv", fc="#e6f4ea")
    box(ax, 0.835, 0.55, 0.15, 0.30,
        "Shunt +\nlow-noise amp\n(on CW305)", fc="#fdf1e3")

    arrow(ax, 0.215, 0.70, 0.255, 0.70)
    ax.text(0.2325, 0.742, "USB", ha="center", fontsize=8, color=MUTE)
    arrow(ax, 0.450, 0.760, 0.585, 0.760)
    ax.text(0.520, 0.775, "5 MHz clock\n(HS2)", ha="center", fontsize=7.6, color=MUTE)
    arrow(ax, 0.450, 0.615, 0.585, 0.615, color=ACC2)
    ax.text(0.5175, 0.578, "trigger in", ha="center", fontsize=7.6, color=ACC2)
    arrow(ax, 0.790, 0.70, 0.835, 0.70)
    arrow(ax, 0.905, 0.55, 0.905, 0.42, color=ACC)
    arrow(ax, 0.905, 0.42, 0.352, 0.42, color=ACC)
    arrow(ax, 0.352, 0.42, 0.352, 0.55, color=ACC)
    ax.text(0.64, 0.445, "analog power signal  ->  ADC, 4 samples per FPGA clock",
            ha="center", fontsize=8.6, color=ACC)

    box(ax, 0.015, 0.06, 0.30, 0.24,
        "What the FPGA does\n676 convolution windows,\none per clock window\n"
        "(8 dwell cycles x 4 ADC samples)", fc="#e6f4ea", fs=8.6)
    box(ax, 0.355, 0.06, 0.30, 0.24,
        "What the ADC returns\n22144 samples per kernel,\n21632 of them useful\n"
        "(676 x 32)", fc="#e8f0fa", fs=8.6)
    box(ax, 0.695, 0.06, 0.285, 0.24,
        "What the host does after\nfeature extraction, template\nlookup, reconstruction,\nscoring",
        fc="#f2f5f8", fs=8.6)
    save(fig, "fig01_signal_chain.png")


# ---------------------------------------------------------------- fig 2
def fig_authenticity():
    fig, ax = plt.subplots(figsize=(11.6, 4.6))
    blank(ax)
    ax.set_title("The gate that failed on 29 Aug and passed on 30 Aug. "
                 "This is why I trust today's traces.", loc="left",
                 fontsize=11.5, weight="bold", pad=12)

    ax.text(0.24, 0.93, "29 August", ha="center", fontsize=11, weight="bold", color=MUTE)
    ax.text(0.76, 0.93, "30 August", ha="center", fontsize=11, weight="bold", color=DEEP)
    ax.plot([0.5, 0.5], [0.03, 0.90], color=MUTE, lw=1.0, ls=":")

    box(ax, 0.02, 0.70, 0.44, 0.16,
        "Read FPGA register pages 2 / 3 / 4\n-> 0x02, 0x05, 0x2e", fc=FILL_SOFT, ec=MUTE)
    box(ax, 0.02, 0.47, 0.44, 0.17,
        "That is the stock AES signature.\nThe board was still running the\nAES example, not my design.",
        fc=FILL_SOFT, ec=MUTE)
    box(ax, 0.02, 0.22, 0.44, 0.19,
        "Any trace captured here would be\nAES power, while the host wrote\nMNIST pixels into AES registers.\n"
        "Capture refused.", fc=FILL_SOFT, ec=MUTE)
    ax.text(0.24, 0.11, "CAPTURE BLOCKED", ha="center", fontsize=11, weight="bold", color=MUTE)

    box(ax, 0.54, 0.70, 0.44, 0.16,
        "Read FPGA register pages 2 / 3 / 4\n-> not the AES signature", fc=FILL_STRONG, ec=DEEP)
    box(ax, 0.54, 0.47, 0.44, 0.17,
        "Write one image + one kernel,\nread back all 676 outputs,\ncompare with the CPU model.",
        fc=FILL_STRONG, ec=DEEP)
    box(ax, 0.54, 0.22, 0.44, 0.19,
        "676 / 676 outputs bit-exact,\n0 mismatches, on every capture\nin this report.",
        fc=FILL_STRONG, ec=DEEP)
    ax.text(0.76, 0.11, "CAPTURE ALLOWED", ha="center", fontsize=11, weight="bold", color=DEEP)

    for y in (0.70, 0.47):
        arrow(ax, 0.24, y, 0.24, y - 0.06, color=MUTE, ls="--")
        arrow(ax, 0.76, y, 0.76, y - 0.06, color=DEEP)
    save(fig, "fig02_authenticity_gate.png")


# ---------------------------------------------------------------- fig 3
def fig_raw_trace():
    d = np.load(os.path.join(HOST, "traces_hard_C1_control", "img5200.npz"))
    tr = d["traces"][0].astype(np.float64)
    spw = int(d["samples_per_window"])
    spc = int(d["samples_per_cycle"])
    dwell = int(d["dwell"])
    freq = 5e6
    adc = freq * spc

    fig = plt.figure(figsize=(11.6, 6.4))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], hspace=0.42, wspace=0.18)

    ax = fig.add_subplot(gs[0, :])
    useful = tr[:N_WINDOWS * spw].reshape(N_WINDOWS, spw)
    w_idx = np.arange(N_WINDOWS)
    ax.fill_between(w_idx, useful.min(axis=1), useful.max(axis=1),
                    color=ACC2, alpha=0.30, lw=0, label="min..max inside the window")
    ax.plot(w_idx, useful.mean(axis=1), lw=0.7, color=INK, label="window mean")
    ax.set_xlabel("convolution window index (each window = 8 FPGA clock cycles)")
    ax.set_ylabel("ADC value (centred)")
    ax.set_title("Real captured power signal, one kernel, one image (img5200, label 4). "
                 "All 676 windows, drawn as the envelope of the 32 samples in each window.",
                 loc="left", fontsize=10.5)
    ax.set_xlim(0, N_WINDOWS - 1)
    ax.legend(fontsize=8, loc="upper right", ncol=2)

    ax2 = fig.add_subplot(gs[1, 0])
    n_show = 4
    seg = tr[:n_show * spw]
    ts = np.arange(seg.size) / adc * 1e6
    ax2.plot(ts, seg, lw=1.1, color=ACC2, marker="o", ms=2.2)
    for w in range(n_show + 1):
        ax2.axvline(w * spw / adc * 1e6, color=ACC, lw=1.0, ls="--")
    for w in range(n_show):
        ax2.text((w + 0.5) * spw / adc * 1e6, ax2.get_ylim()[1] * 0.86,
                 f"window {w}", ha="center", fontsize=8, color=ACC)
    ax2.set_xlabel("time (microseconds)")
    ax2.set_ylabel("ADC value")
    ax2.set_title(f"Zoom: {spw} samples per window "
                  f"({dwell} FPGA cycles x {spc} ADC samples)", loc="left", fontsize=10)

    ax3 = fig.add_subplot(gs[1, 1])
    feats = window_features(tr, spw)
    ax3.plot(feats[:120], lw=1.0, color=INK)
    ax3.set_xlabel("convolution window index")
    ax3.set_ylabel("mean |power| in window")
    ax3.set_title("The one number kept per window (first 120 of 676)",
                  loc="left", fontsize=10)
    save(fig, "fig03_raw_trace.png")


# ---------------------------------------------------------------- fig 4
def fig_feature_map():
    d = np.load(os.path.join(HOST, "traces_hard_C1_control", "img5200.npz"))
    tr = d["traces"].astype(np.float64)
    spw = int(d["samples_per_window"])
    img = d["image"]
    feats = np.stack([window_features(tr[k], spw) for k in range(tr.shape[0])])

    fig = plt.figure(figsize=(11.6, 5.6))
    gs = fig.add_gridspec(2, 4, hspace=0.42, wspace=0.30, height_ratios=[1.25, 1])

    grey = np.load(os.path.join(HOST, "mnist_test.npz"))["images"][5200]

    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(grey, cmap="gray_r")
    ax.set_title("original MNIST image\n(0..255, never reaches the FPGA)", fontsize=9.0)
    ax.axis("off")

    ax = fig.add_subplot(gs[0, 1])
    ax.imshow(img, cmap="gray_r", vmin=0, vmax=1)
    ax.set_title("what the FPGA receives\n(binarised at 127; also the\nground truth I score against)",
                 fontsize=9.0)
    ax.axis("off")

    for i, k in enumerate([0, 4]):
        ax = fig.add_subplot(gs[0, 2 + i])
        fm = feats[k].reshape(OUT_SIDE, OUT_SIDE)
        ax.imshow(fm, cmap="magma")
        ax.set_title(f"power feature map\nkernel {k}  (26x26)", fontsize=9.5)
        ax.axis("off")

    ax = fig.add_subplot(gs[1, :])
    z = (feats - np.median(feats, axis=1, keepdims=True))
    z = z / (1.4826 * np.median(np.abs(z), axis=1, keepdims=True) + 1e-9)
    im = ax.imshow(z, aspect="auto", cmap="viridis", vmin=-3, vmax=3)
    ax.set_xlabel("convolution window index (0 .. 675)")
    ax.set_ylabel("probe kernel")
    ax.set_title("The power vector the attack actually uses: 9 kernels x 676 windows, "
                 "normalised per kernel", loc="left", fontsize=10)
    fig.colorbar(im, ax=ax, pad=0.01, fraction=0.03, label="normalised power")

    fig.suptitle("From the captured signal to the attack feature. "
                 "The digit is already visible in the raw power of a single kernel.",
                 fontsize=11.5, weight="bold", x=0.012, ha="left", y=1.005)
    save(fig, "fig04_feature_map.png")


# ---------------------------------------------------------------- fig 5
def fig_algorithm():
    fig, ax = plt.subplots(figsize=(11.6, 6.6))
    blank(ax)
    ax.set_title("The active power-template attack, as I implemented it.",
                 loc="left", fontsize=11.5, weight="bold", pad=12)

    ax.text(0.012, 0.885, "PROFILING  (attacker owns a board, images are known)",
            fontsize=9.6, weight="bold", color=ACC2)
    box(ax, 0.012, 0.70, 0.20, 0.14,
        "40 known images\ncaptured on the CW305", fc=FILL_MID)
    box(ax, 0.245, 0.70, 0.21, 0.14,
        "for every window:\n9-kernel power vector", fc=FILL_MID)
    box(ax, 0.488, 0.70, 0.22, 0.14,
        "label it with the true\n3x3 binary patch\n(512 possible patches)", fc=FILL_MID)
    box(ax, 0.742, 0.70, 0.24, 0.14,
        "template:\n40 x 676 = 27040 rows\nof (power vector, patch)", fc=FILL_STRONG)
    arrow(ax, 0.212, 0.77, 0.245, 0.77)
    arrow(ax, 0.455, 0.77, 0.488, 0.77)
    arrow(ax, 0.710, 0.77, 0.742, 0.77)

    ax.text(0.012, 0.60, "ATTACK  (only traces, image never used)",
            fontsize=9.6, weight="bold", color=DEEP)
    box(ax, 0.012, 0.41, 0.20, 0.14,
        "unknown image,\ntrace-only bundle", fc=FILL_SOFT)
    box(ax, 0.245, 0.41, 0.21, 0.14,
        "same feature\nextraction", fc=FILL_SOFT)
    box(ax, 0.488, 0.41, 0.22, 0.14,
        "split 9 kernels into\n3 groups of 3,\nkD-tree search, delta=1.0", fc=FILL_SOFT)
    box(ax, 0.742, 0.41, 0.24, 0.14,
        "intersect the 3 groups\n-> candidate patches\nper window", fc=FILL_MID)
    arrow(ax, 0.212, 0.48, 0.245, 0.48, color=INK)
    arrow(ax, 0.455, 0.48, 0.488, 0.48, color=INK)
    arrow(ax, 0.710, 0.48, 0.742, 0.48, color=INK)
    arrow(ax, 0.862, 0.70, 0.862, 0.55, color=ACC2, ls="--")
    ax.text(0.872, 0.625, "template is queried", fontsize=8, color=ACC2, va="center")

    ax.text(0.012, 0.31, "RECONSTRUCTION", fontsize=9.6, weight="bold", color=DEEP)
    box(ax, 0.012, 0.12, 0.30, 0.15,
        "greedy selection\n(Algorithm 2 of the paper):\npick the candidate that agrees\n"
        "best with neighbours", fc=FILL_MID)
    box(ax, 0.355, 0.12, 0.28, 0.15,
        "overlapping 3x3 patches\nvote on every pixel;\naverage the votes", fc=FILL_MID)
    box(ax, 0.675, 0.12, 0.31, 0.15,
        "recovered 28x28 image\n-> F1, MSSIM, pixel distance,\ndigit recogniser", fc=FILL_STRONG)
    arrow(ax, 0.312, 0.195, 0.355, 0.195, color=INK)
    arrow(ax, 0.635, 0.195, 0.675, 0.195, color=INK)
    arrow(ax, 0.862, 0.41, 0.862, 0.28, color=INK)
    save(fig, "fig05_algorithm.png")


# ---------------------------------------------------------------- fig 6
def fig_geometry():
    fig, axes = plt.subplots(1, 3, figsize=(11.6, 3.9))
    fig.suptitle("Where my hardware layer differs from the layer in the paper.",
                 fontsize=11.5, weight="bold", x=0.012, ha="left", y=1.04)

    grey = np.load(os.path.join(HOST, "mnist_test.npz"))["images"][5200]
    binar = (grey > 127).astype(np.uint8)

    ax = axes[0]
    ax.imshow(grey, cmap="gray_r", vmin=0, vmax=255)
    ax.add_patch(Rectangle((-0.5, -0.5), 28, 28, fill=False, ec=ACC2, lw=2.2))
    ax.set_title("paper: real 0..255 pixels,\n'same' padding, 28x28 out, 784 cycles",
                 fontsize=9.3)
    ax.axis("off")

    ax = axes[1]
    ax.imshow(binar, cmap="gray_r", vmin=0, vmax=1)
    # the ring of pixels a valid convolution never produces an output for
    ax.add_patch(Rectangle((-0.5, -0.5), 28, 28, fill=False, ec=MUTE, lw=1.4, ls="--"))
    ax.add_patch(Rectangle((0.5, 0.5), 26, 26, fill=False, ec=ACC, lw=2.2))
    ax.annotate("no output for this 1-pixel border", xy=(13.5, 27.2), xytext=(13.5, 31.8),
                ha="center", fontsize=7.8, color=ACC,
                arrowprops=dict(arrowstyle="->", color=ACC, lw=1.1))
    ax.set_title("mine: binarised pixels,\nvalid conv, 26x26 out, 676 windows",
                 fontsize=9.3)
    ax.axis("off")

    ax = axes[2]
    blank(ax)
    ax.text(0.0, 0.95, "Consequence", fontsize=10, weight="bold", va="top")
    ax.text(0.0, 0.80,
            "The paper recovers grey levels, because\n"
            "its first layer multiplies real pixels by\n"
            "binary weights.\n\n"
            "My FPGA layer thresholds the pixel first,\n"
            "so the most I can recover is a 1-bit\n"
            "silhouette. I therefore score with F1,\n"
            "IoU and MSSIM, and I always print the\n"
            "all-background baseline next to pixel\n"
            "accuracy.",
            fontsize=9.0, va="top", linespacing=1.6)
    save(fig, "fig06_geometry.png")


# ---------------------------------------------------------------- fig 7
def fig_sweep():
    rows = []
    for r in range(1, 6):
        p = os.path.join(HT, f"score_sweep_R{r}.json")
        if os.path.exists(p):
            rows.append((r, load_json(p)["metrics"]))
    if not rows:
        print("skip sweep figure: no scores yet")
        return
    R = [r for r, _ in rows]
    ba = [m["bit_acc"] for _, m in rows]
    az = [m["all_zero_bit_acc"] for _, m in rows]
    f1 = [m["foreground_f1"] for _, m in rows]
    ms = [m["mssim"] for _, m in rows]
    rec = [m["recognition_accuracy_recovered"] for _, m in rows]

    fig, axes = plt.subplots(1, 3, figsize=(11.6, 3.7))
    fig.suptitle("How many hardware repeats the attack needs. One capture at "
                 "avg=5, subset afterwards, so images and board state are identical.",
                 fontsize=11.5, weight="bold", x=0.012, ha="left", y=1.06)

    ax = axes[0]
    ax.plot(R, ba, "o-", color=ACC2, label="recovered")
    ax.plot(R, az, "s--", color=MUTE, label="all-background baseline")
    ax.fill_between(R, az, ba, color=ACC2, alpha=0.12)
    ax.set_xlabel("averaged hardware repeats")
    ax.set_ylabel("pixel accuracy")
    ax.set_title("pixel accuracy vs the honest baseline", fontsize=9.8)
    ax.set_xticks(R)
    ax.legend(fontsize=8, loc="lower right")

    ax = axes[1]
    ax.plot(R, f1, "o-", color=GOOD, label="foreground F1")
    ax.plot(R, ms, "^-", color=WARN, label="MSSIM")
    ax.set_xlabel("averaged hardware repeats")
    ax.set_title("the metrics that actually move", fontsize=9.8)
    ax.set_xticks(R)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8, loc="lower right")

    ax = axes[2]
    ax.plot(R, rec, "o-", color=ACC)
    ax.set_xlabel("averaged hardware repeats")
    ax.set_ylabel("digit recogniser accuracy")
    ax.set_title("recognition of the recovered image", fontsize=9.8)
    ax.set_xticks(R)
    ax.set_ylim(0, 1.05)
    save(fig, "fig07_repeat_sweep.png")


# ---------------------------------------------------------------- fig 8
def fig_conditions():
    order = [("C1_control", "same conditions\n(control)"),
             ("C2_reprogram", "FPGA\nreprogrammed"),
             ("C3_clk7m5", "clock\n5 -> 7.5 MHz"),
             ("C4_clk10m", "clock\n5 -> 10 MHz"),
             ("C5_gain30", "gain\n40 -> 30 dB"),
             ("C6_gain50", "gain\n40 -> 50 dB")]
    rows = []
    for key, lab in order:
        p = os.path.join(HT, f"score_cond_{key}.json")
        if os.path.exists(p):
            rows.append((lab, load_json(p)["metrics"]))
    if not rows:
        print("skip condition figure: no scores yet")
        return
    labs = [l for l, _ in rows]
    f1 = [m["foreground_f1"] for _, m in rows]
    ba = [m["bit_acc"] for _, m in rows]
    az = [m["all_zero_bit_acc"] for _, m in rows]
    rec = [m["recognition_accuracy_recovered"] for _, m in rows]
    x = np.arange(len(rows))

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.1))
    fig.suptitle("Template transfer. One profile, built hours earlier at "
                 "5 MHz / 40 dB, attacking the same 40 images under changed conditions.",
                 fontsize=11.5, weight="bold", x=0.012, ha="left", y=1.03)

    ax = axes[0]
    # one colour for every bar: the height already says which condition failed,
    # and colouring by value re-encodes the same thing as a traffic light
    ax.bar(x, f1, color=ACC2, width=0.62)
    for i, v in enumerate(f1):
        ax.text(i, v + 0.015, f"{v:.2f}", ha="center", fontsize=8.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labs, fontsize=7.6)
    ax.set_ylabel("foreground F1")
    ax.set_ylim(0, 1)
    ax.set_title("foreground F1 by condition", fontsize=10)

    ax = axes[1]
    ax.bar(x - 0.19, ba, width=0.36, color=ACC2, label="pixel accuracy")
    ax.bar(x + 0.19, az, width=0.36, color=MUTE, label="all-background baseline")
    ax.plot(x, rec, "o--", color=ACC, label="recogniser accuracy")
    ax.set_xticks(x)
    ax.set_xticklabels(labs, fontsize=7.6)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8, loc="lower left")
    ax.set_title("pixel accuracy against its baseline, and recognition", fontsize=10)
    save(fig, "fig08_conditions.png")


# ---------------------------------------------------------------- fig 9
def fig_examples():
    combos = [("cond_results_C1_control", "cond_truth_C1_control.json", "same conditions"),
              ("cond_results_C2_reprogram", "cond_truth_C2_reprogram.json", "after reprogram"),
              ("cond_results_C4_clk10m", "cond_truth_C4_clk10m.json", "clock 10 MHz"),
              ("cond_results_C5_gain30", "cond_truth_C5_gain30.json", "gain 30 dB")]
    avail = [(r, t, l) for r, t, l in combos
             if os.path.isdir(os.path.join(HT, r)) and os.path.exists(os.path.join(HT, t))]
    if not avail:
        print("skip examples figure")
        return
    n_img = 6
    fig, axes = plt.subplots(len(avail) + 1, n_img, figsize=(1.35 * n_img, 1.42 * (len(avail) + 1)))
    truth = load_json(os.path.join(HT, avail[0][1]))
    items = truth["items"][:n_img]
    def bare(ax):
        """Keep the axes (so ylabel stays anchored to its own row) but hide ticks."""
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(False)

    for j, it in enumerate(items):
        axes[0, j].imshow(np.asarray(it["image"], dtype=np.uint8), cmap="gray_r",
                          vmin=0, vmax=1)
        bare(axes[0, j])
        axes[0, j].set_title(f"digit {it['label']}", fontsize=8.5)
    axes[0, 0].set_ylabel("truth", fontsize=8.6, weight="bold")

    for i, (rdir, tjson, lab) in enumerate(avail):
        for j, it in enumerate(items):
            idx = it["file"][3:7]
            p = os.path.join(HT, rdir, f"recovered_{idx}.npz")
            ax = axes[i + 1, j]
            if os.path.exists(p):
                ax.imshow(np.load(p)["recovered"], cmap="gray_r", vmin=0, vmax=1)
            bare(ax)
        axes[i + 1, 0].set_ylabel(lab, fontsize=8.6, weight="bold")
    fig.suptitle("The same six images recovered under each condition.",
                 fontsize=11.5, weight="bold", x=0.02, ha="left", y=1.005)
    fig.subplots_adjust(left=0.085, wspace=0.06, hspace=0.10)
    save(fig, "fig09_examples.png")


# ---------------------------------------------------------------- fig 10
def fig_why_stalled():
    fig, ax = plt.subplots(figsize=(11.6, 4.9))
    blank(ax)
    ax.set_title("Three separate reasons the earlier attempts looked weak. "
                 "Only the first one is now fixed.", loc="left",
                 fontsize=11.5, weight="bold", pad=12)

    box(ax, 0.012, 0.60, 0.30, 0.30,
        "1. Wrong design on the FPGA\n\nThe board answered as the stock\nAES example. The host wrote\n"
        "MNIST pixels into AES registers,\nso the traces were AES power.",
        fc=FILL_STRONG, ec=DEEP, fs=8.8)
    ax.text(0.162, 0.535, "FIXED  (verified 30 Aug)", ha="center", fontsize=9,
            weight="bold", color=DEEP)

    box(ax, 0.35, 0.60, 0.30, 0.30,
        "2. Signal dilution\n\nThe conv unit is small next to the\nrest of the fabric. Its power is\n"
        "buried under activity that varies\nfrom trace to trace.",
        fc=FILL_MID, ec=STEEL, fs=8.8)
    ax.text(0.50, 0.535, "STILL THE LIMIT", ha="center", fontsize=9,
            weight="bold", color=STEEL)

    box(ax, 0.688, 0.60, 0.30, 0.30,
        "3. The old write-up blamed the\n    wrong thing\n\nFINAL_RESULTS.md blamed per-cycle\n"
        "extraction and smearing. That was\nnot the cause.",
        fc=FILL_SOFT, ec=MUTE, fs=8.8)
    ax.text(0.838, 0.535, "CORRECTED HERE", ha="center", fontsize=9,
            weight="bold", color=MUTE)

    box(ax, 0.012, 0.10, 0.976, 0.33,
        "What this means in practice\n\n"
        "Averaging is the knob that buys back the diluted signal: with one hardware repeat the attack is "
        "barely above the all-background\nbaseline, and with five repeats the foreground F1 grows by about 60%. "
        "So the earlier 'low detection' was two problems stacked -\nthe traces were from the wrong design, "
        "and even correct traces need averaging before the template becomes usable.",
        fc="#f2f5f8", fs=9.2)
    save(fig, "fig10_why_stalled.png")


# ---------------------------------------------------------------- fig 11
def fig_noise_vs_repeats():
    d = np.load(os.path.join(HOST, "traces_hard_C1_control", "img5200.npz"))
    reps = d["repeats"][0].astype(np.float64)
    spw = int(d["samples_per_window"])
    img = d["image"].astype(np.uint8)  # already binarised by the capture script

    fig, axes = plt.subplots(1, 6, figsize=(11.6, 2.45))
    fig.suptitle("Why averaging matters, shown on the raw signal: the power "
                 "feature map of one kernel as repeats are added.",
                 fontsize=11.5, weight="bold", x=0.012, ha="left", y=1.09)
    for i, r in enumerate([1, 2, 3, 4, 5]):
        f = window_features(reps[:r].mean(axis=0), spw).reshape(OUT_SIDE, OUT_SIDE)
        ax = axes[i]
        ax.imshow(f, cmap="magma")
        ax.set_title(f"{r} repeat" + ("" if r == 1 else "s"), fontsize=9)
        ax.axis("off")
    axes[5].imshow(img, cmap="gray_r", vmin=0, vmax=1)
    axes[5].set_title("binarised truth", fontsize=9)
    axes[5].axis("off")
    save(fig, "fig11_noise_vs_repeats.png")


# ---------------------------------------------------------------- fig 12
def fig_comparison_table():
    rows = [
        ("", "Wei et al. (ACSAC'18)", "This work"),
        ("FPGA board", "SAKURA-G, Spartan-6 LX75", "CW305, Artix-7 XC7A100T"),
        ("Oscilloscope", "Tektronix MDO3034, 2.5 GS/s,\nasynchronous",
         "ChipWhisperer-Lite, synchronous,\n4 samples per FPGA clock"),
        ("Alignment", "DC restoration + per-cycle\ncurve fitting (their section 5)",
         "not needed: the ADC is locked\nto the FPGA clock"),
        ("Attacked layer", "first conv, 28x28 'same',\nreal 0..255 pixels",
         "first conv, 26x26 valid,\nbinarised pixels"),
        ("Cycles per image", "784", "676 windows of 8 cycles"),
        ("Probe kernels", "9 of 64", "9 of 64"),
        ("Template / eval images", "300 / 200", "40 / 40 here, 300 / 200 at\npaper scale"),
        ("Reported result", "89.8 % recognition (3x3)",
         "see the results section;\nbaseline always printed"),
    ]
    fig, ax = plt.subplots(figsize=(11.6, 4.6))
    blank(ax)
    ax.set_title("My setup next to the paper's setup.", loc="left",
                 fontsize=11.5, weight="bold", pad=10)
    y = 0.93
    h = 0.098
    for i, (a, b, c) in enumerate(rows):
        head = (i == 0)
        fc = "#dbe7f6" if head else ("#f7f9fb" if i % 2 else "white")
        for x, w, txt, al in ((0.008, 0.20, a, "left"), (0.215, 0.38, b, "left"),
                              (0.602, 0.39, c, "left")):
            ax.add_patch(Rectangle((x, y - h), w, h, facecolor=fc, edgecolor="#c8d2dc", lw=0.8))
            ax.text(x + 0.012, y - h / 2, txt, ha=al, va="center",
                    fontsize=8.4, weight="bold" if head else "normal", linespacing=1.35)
        y -= h
    save(fig, "fig12_comparison.png")


# ---------------------------------------------------------------- fig 13
def fig_clock_mixing():
    """Does profiling under several clocks buy tolerance?  Three templates, all
    built from 40 profiling images, each attacked at both clocks."""
    templates = [("5m", "profiled at\n5 MHz only", ACC2),
                 ("10m", "profiled at\n10 MHz only", GOOD),
                 ("mixed", "mixed 20+20", WARN)]
    attacks = [("C1_control", "attack at 5 MHz"), ("C4_clk10m", "attack at 10 MHz")]
    data = {}
    for tkey, _, _ in templates:
        for akey, _ in attacks:
            p = os.path.join(HT, f"score_mix_{tkey}_vs_{akey}.json")
            if os.path.exists(p):
                data[(tkey, akey)] = load_json(p)["metrics"]
    if len(data) < len(templates) * len(attacks):
        print("skip clock-mixing figure: matrix incomplete")
        return

    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.2))
    fig.suptitle("Profiling under several clocks. Every template uses 40 profiling "
                 "images and attacks the same 40 evaluation images.",
                 fontsize=11.5, weight="bold", x=0.012, ha="left", y=1.03)

    x = np.arange(len(attacks))
    width = 0.26
    ax = axes[0]
    for i, (tkey, lab, col) in enumerate(templates):
        vals = [data[(tkey, akey)]["foreground_f1"] for akey, _ in attacks]
        bars = ax.bar(x + (i - 1) * width, vals, width, color=col, label=lab.replace("\n", " "))
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.2f}",
                    ha="center", fontsize=8.2)
    base = data[("5m", "C1_control")]["all_zero_bit_acc"]
    ax.set_xticks(x)
    ax.set_xticklabels([l for _, l in attacks])
    ax.set_ylabel("foreground F1")
    ax.set_ylim(0, 1)
    ax.set_title("the 5 MHz template is the only one that collapses", fontsize=10)
    ax.legend(fontsize=8, loc="upper right")

    ax = axes[1]
    for i, (tkey, lab, col) in enumerate(templates):
        vals = [data[(tkey, akey)]["recognition_accuracy_recovered"] for akey, _ in attacks]
        bars = ax.bar(x + (i - 1) * width, vals, width, color=col)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.2f}",
                    ha="center", fontsize=8.2)
    ax.set_xticks(x)
    ax.set_xticklabels([l for _, l in attacks])
    ax.set_ylabel("digit recogniser accuracy")
    ax.set_ylim(0, 1.08)
    ax.set_title("recognition of the recovered image", fontsize=10)
    save(fig, "fig13_clock_mixing.png")


# ---------------------------------------------------------------- fig 14
def fig_grey_recovery():
    """The point of removing the binarisation: are grey levels recoverable?"""
    grey_dir = os.path.join(HT, "p2p_grey_e60")
    bin_dir = os.path.join(HT, "p2p_matched_e60")
    gp = os.path.join(grey_dir, "test_recovered.npz")
    if not os.path.exists(gp):
        print("skip grey recovery figure: no grey p2p result yet")
        return
    def truth_from_capture(result_dir, saved):
        """Read the ground truth from the capture itself.

        The saved test_recovered.npz is not trusted for this: earlier runs cast the
        truth to uint8, which floors a grey truth in [0,1] to zero. The capture
        shards always hold the real pixels, and summary.json records which image
        indices were in the test split."""
        import glob as _glob
        summ = os.path.join(result_dir, "summary.json")
        if not os.path.exists(summ):
            return saved
        meta = load_json(summ)
        trace_dir = meta.get("trace_dir")
        want = meta.get("test_indices")
        if not trace_dir or not want or not os.path.isdir(trace_dir):
            return saved
        idx, imgs = [], []
        for shard in sorted(_glob.glob(os.path.join(trace_dir, "shard*.npz"))):
            d = np.load(shard)
            idx.append(d["indices"])
            imgs.append(d["images"])
        if not idx:
            return saved
        idx = np.concatenate(idx)
        imgs = np.concatenate(imgs).astype(np.float32)
        if imgs.max() > 1.5:                      # grey capture, 0..255
            imgs /= 255.0
        pos = {int(v): i for i, v in enumerate(idx)}
        rows = [pos[int(v)] for v in want if int(v) in pos]
        return imgs[rows] if len(rows) == len(want) else saved

    g = np.load(gp)
    grec = g["recovered"]
    gtru = truth_from_capture(grey_dir, g["truth"].astype(np.float32))

    have_bin = os.path.exists(os.path.join(bin_dir, "test_recovered.npz"))
    if have_bin:
        b = np.load(os.path.join(bin_dir, "test_recovered.npz"))
        brec = b["recovered"]
        btru = truth_from_capture(bin_dir, b["truth"].astype(np.float32))

    n = 8
    rows = 4 if have_bin else 2
    fig, axes = plt.subplots(rows, n, figsize=(1.28 * n, 1.34 * rows))

    def strip(ax):
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(False)

    for j in range(n):
        axes[0, j].imshow(gtru[j], cmap="gray_r", vmin=0, vmax=1)
        strip(axes[0, j])
        axes[1, j].imshow(np.clip(grec[j], 0, 1), cmap="gray_r", vmin=0, vmax=1)
        strip(axes[1, j])
    axes[0, 0].set_ylabel("grey\ntruth", fontsize=8.4, weight="bold")
    axes[1, 0].set_ylabel("grey\nrecovered", fontsize=8.4, weight="bold")

    if have_bin:
        for j in range(n):
            axes[2, j].imshow(btru[j], cmap="gray_r", vmin=0, vmax=1)
            strip(axes[2, j])
            axes[3, j].imshow(np.clip(brec[j], 0, 1), cmap="gray_r", vmin=0, vmax=1)
            strip(axes[3, j])
        axes[2, 0].set_ylabel("binary\ntruth", fontsize=8.4, weight="bold")
        axes[3, 0].set_ylabel("binary\nrecovered", fontsize=8.4, weight="bold")

    fig.suptitle("Removing the input binarisation. The grey design recovers real grey "
                 "levels, not only a silhouette.",
                 fontsize=11.5, weight="bold", x=0.02, ha="left", y=1.005)
    fig.subplots_adjust(left=0.085, wspace=0.06, hspace=0.08)
    save(fig, "fig14_grey_recovery.png")


# ---------------------------------------------------------------- fig 15
def fig_delta_sweep():
    """F1 against the candidate-search radius, for two template sizes."""
    import glob
    series = {}
    for f in glob.glob(os.path.join(HT, "score_delta_*.json")):
        base = os.path.basename(f)[len("score_delta_"):-len(".json")]
        tag, d = base.rsplit("_d", 1)
        series.setdefault(tag, []).append((float(d.replace("p", ".")),
                                           load_json(f)["metrics"]))
    if not series:
        print("skip delta figure: no sweep scores")
        return
    for k in series:
        series[k].sort(key=lambda r: r[0])

    styles = {"p40": ("40 profiling images", ACC2, "o-"),
              "p300": ("300 profiling images", DEEP, "s-")}

    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.2))
    fig.suptitle("The search radius is not a constant of the attack. Published value "
                 "$\\delta$ = 1.0 is far from optimal on this hardware.",
                 fontsize=11.5, weight="bold", x=0.012, ha="left", y=1.03)

    ax = axes[0]
    for tag, (lab, col, mk) in styles.items():
        if tag not in series:
            continue
        xs = [d for d, _ in series[tag]]
        ys = [m["foreground_f1"] for _, m in series[tag]]
        ax.plot(xs, ys, mk, color=col, label=lab, ms=4.2, lw=1.5)
        bi = int(np.argmax(ys))
        ax.plot([xs[bi]], [ys[bi]], "o", ms=11, mfc="none", mec=col, mew=1.8)
        ax.annotate(f"$\\delta$={xs[bi]:g}\n{ys[bi]:.3f}", (xs[bi], ys[bi]),
                    textcoords="offset points", xytext=(6, -26),
                    fontsize=8.2, color=col)
    ax.axvline(1.0, color=MUTE, ls=":", lw=1.2)
    ax.text(1.06, 0.985, "published\n$\\delta$ = 1.0", fontsize=8, color=MUTE,
            ha="left", va="top", transform=ax.get_xaxis_transform())
    ax.set_xscale("log")
    ax.set_xlabel("candidate search radius $\\delta$ (log scale)")
    ax.set_ylabel("foreground F1")
    ax.set_title("both curves peak well below the published value", fontsize=10)
    ax.legend(fontsize=8.4, loc="lower center")

    ax = axes[1]
    for tag, (lab, col, mk) in styles.items():
        if tag not in series:
            continue
        xs = [d for d, _ in series[tag]]
        ys = [m["recognition_accuracy_recovered"] for _, m in series[tag]]
        ax.plot(xs, ys, mk, color=col, label=lab, ms=4.2, lw=1.5)
    ax.axvline(1.0, color=MUTE, ls=":", lw=1.2)
    ax.set_xscale("log")
    ax.set_xlabel("candidate search radius $\\delta$ (log scale)")
    ax.set_ylabel("digit recogniser accuracy")
    ax.set_ylim(0, 1.06)
    ax.set_title("recognition of the recovered image", fontsize=10)
    save(fig, "fig15_delta_sweep.png")


if __name__ == "__main__":
    fig_signal_chain()
    fig_authenticity()
    fig_raw_trace()
    fig_feature_map()
    fig_algorithm()
    fig_geometry()
    fig_sweep()
    fig_conditions()
    fig_examples()
    fig_why_stalled()
    fig_noise_vs_repeats()
    fig_comparison_table()
    fig_clock_mixing()
    fig_grey_recovery()
    fig_delta_sweep()
    print("done")
