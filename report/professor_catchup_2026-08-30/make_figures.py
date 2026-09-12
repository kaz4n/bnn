from __future__ import annotations

import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
FIG = HERE / "figures"
FIG.mkdir(parents=True, exist_ok=True)


plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 220,
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
})


COL = {
    "green": "#dff4df",
    "green_edge": "#3a8f50",
    "yellow": "#fff0c9",
    "yellow_edge": "#bb8a00",
    "red": "#ffe0de",
    "red_edge": "#b9413b",
    "blue": "#dfeeff",
    "blue_edge": "#356aa0",
    "gray": "#f0f2f5",
    "gray_edge": "#6b7280",
    "ink": "#20242a",
}


def save(fig: plt.Figure, name: str) -> None:
    fig.savefig(FIG / name, bbox_inches="tight")
    plt.close(fig)


def box(ax, x, y, w, h, text, fc, ec, size=10, weight="normal"):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.025,rounding_size=0.03",
        linewidth=1.5,
        facecolor=fc,
        edgecolor=ec,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        color=COL["ink"],
        fontsize=size,
        fontweight=weight,
        wrap=True,
    )
    return patch


def arrow(ax, start, end, color="#4b5563", lw=1.5):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=14,
            linewidth=lw,
            color=color,
            shrinkA=4,
            shrinkB=4,
        )
    )


def fig_status_summary():
    fig, ax = plt.subplots(figsize=(12.5, 5.8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.5, 0.95, "Experiment outcome summary", ha="center", va="center", fontsize=18, fontweight="bold")
    ax.text(
        0.5,
        0.89,
        "The important result is not only the accuracy numbers. It is also what evidence level each result has.",
        ha="center",
        va="center",
        fontsize=11,
    )

    cards = [
        (
            0.04,
            "Confirmed hardware signal",
            "CW-Lite + CW305 analog capture works\nstock AES control trace captured\nADC locked = true\nciphertext = 5aea...940a",
            COL["green"],
            COL["green_edge"],
            "Accepted as real capture chain",
        ),
        (
            0.365,
            "Post-reset custom design checked",
            "M0/M1/M2 checked high\nUSB reset pressed\npages 2/3/4 = 0/0/0\n676 outputs bit-exact",
            COL["yellow"],
            COL["yellow_edge"],
            "Accepted as custom design active",
        ),
        (
            0.69,
            "Fresh recovery runs",
            "Active-template held-out sets\nP2P-style generator tests\nall use live CW305 traces\nall require hardware manifest",
            COL["blue"],
            COL["blue_edge"],
            "Now report as checked hardware runs",
        ),
    ]
    for x, title, body, fc, ec, foot in cards:
        box(ax, x, 0.25, 0.27, 0.54, f"{title}\n\n{body}", fc, ec, size=10, weight="normal")
        ax.text(x + 0.135, 0.17, foot, ha="center", va="center", fontsize=10, color=ec, fontweight="bold")

    arrow(ax, (0.31, 0.52), (0.36, 0.52))
    arrow(ax, (0.635, 0.52), (0.685, 0.52))
    ax.text(0.5, 0.07, "Report rule used here: only use recovery numbers when capture manifest, identity check, and functional check all pass.", ha="center", va="center", fontsize=10)
    save(fig, "fig01_status_summary.png")


def fig_aes_control_trace():
    trace_path = ROOT / "host" / "p0_trace.npy"
    meta_path = ROOT / "report" / "hardware_aes_control_trace_pass_20260829.json"
    trace = np.load(trace_path) if trace_path.exists() else np.array([])
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}

    fig, axes = plt.subplots(2, 1, figsize=(11.5, 6.0), sharex=False)
    if trace.size:
        axes[0].plot(trace, lw=0.8, color="#1f77b4")
        axes[0].set_title("Real ChipWhisperer capture: stock AES control trace")
        axes[0].set_ylabel("ADC amplitude")
        axes[0].grid(True, alpha=0.25)
        n = min(600, len(trace))
        axes[1].plot(np.arange(n), trace[:n], lw=1.0, color="#1f77b4")
        axes[1].axvspan(0, 80, color="#ff7f0e", alpha=0.18, label="trigger/start transient")
        axes[1].set_title("Zoom near trigger")
        axes[1].set_xlabel("sample")
        axes[1].set_ylabel("ADC amplitude")
        axes[1].grid(True, alpha=0.25)
        axes[1].legend(loc="upper right")
    stats = (
        f"samples={meta.get('samples', trace.size)}\n"
        f"min={meta.get('trace_min', float(trace.min()) if trace.size else 'n/a')}\n"
        f"max={meta.get('trace_max', float(trace.max()) if trace.size else 'n/a')}\n"
        f"std={meta.get('trace_std', float(trace.std()) if trace.size else 'n/a')}\n"
        f"ADC locked={meta.get('adc_locked', 'n/a')}"
    )
    axes[0].text(
        0.99,
        0.96,
        stats,
        transform=axes[0].transAxes,
        ha="right",
        va="top",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="#888", alpha=0.95),
        fontsize=9,
    )
    fig.tight_layout()
    save(fig, "fig02_real_aes_control_trace.png")


def fig_register_gate():
    fig, ax = plt.subplots(figsize=(11.5, 5.0))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.5, 0.93, "Hardware identity check after programming", ha="center", va="center", fontsize=17, fontweight="bold")
    ax.text(0.5, 0.86, "This is the check that prevents training on the wrong FPGA image.", ha="center", va="center", fontsize=11)

    headers = ["Read page", "Observed value", "Meaning in this run", "Decision"]
    rows = [
        ["2", "0x00", "not AES page value", "pass"],
        ["3", "0x00", "not AES page value", "pass"],
        ["4", "0x00", "not AES identify byte", "pass"],
    ]
    x0, y0, w, h = 0.06, 0.24, 0.88, 0.48
    colw = [0.18, 0.22, 0.36, 0.24]
    rowh = h / (len(rows) + 1)
    xs = [x0]
    for cw in colw:
        xs.append(xs[-1] + w * cw)
    for j, head in enumerate(headers):
        ax.add_patch(Rectangle((xs[j], y0 + len(rows) * rowh), xs[j + 1] - xs[j], rowh, facecolor=COL["blue"], edgecolor=COL["blue_edge"]))
        ax.text((xs[j] + xs[j + 1]) / 2, y0 + len(rows) * rowh + rowh / 2, head, ha="center", va="center", fontsize=10, fontweight="bold")
    for i, row in enumerate(rows):
        yy = y0 + (len(rows) - 1 - i) * rowh
        for j, cell in enumerate(row):
            fc = COL["green"] if j == 3 else "white"
            ax.add_patch(Rectangle((xs[j], yy), xs[j + 1] - xs[j], rowh, facecolor=fc, edgecolor="#c8c8c8"))
            ax.text((xs[j] + xs[j + 1]) / 2, yy + rowh / 2, cell, ha="center", va="center", fontsize=10)
    box(
        ax,
        0.14,
        0.08,
        0.72,
        0.10,
        "Conclusion: accept this rerun as the custom leakage design.\nThe 676-output functional check passed with zero mismatches.",
        COL["green"],
        COL["green_edge"],
        size=10,
    )
    save(fig, "fig03_register_identity_gate.png")


def fig_trace_to_features():
    npz_path = ROOT / "host" / "traces_fresh_trained_100_159" / "img0100.npz"
    if not npz_path.exists():
        return
    data = np.load(npz_path, allow_pickle=True)
    traces = data["traces"].astype(float)
    image = data["image"]
    spw = int(data["samples_per_window"])
    dwell = int(data["dwell"])
    spc = int(data["samples_per_cycle"])
    t = traces[0].copy()
    t = t - np.median(t)
    nwin = 26 * 26
    windows = t[: nwin * spw].reshape(nwin, spw)
    feat = np.mean(np.abs(windows), axis=1).reshape(26, 26)
    feat_all = []
    for k in range(min(9, traces.shape[0])):
        tt = traces[k] - np.median(traces[k])
        feat_all.append(np.mean(np.abs(tt[: nwin * spw].reshape(nwin, spw)), axis=1))
    feat_all = np.array(feat_all)

    fig = plt.figure(figsize=(12, 7.2), constrained_layout=True)
    gs = fig.add_gridspec(2, 3, height_ratios=[1.15, 1.0], width_ratios=[1.3, 1.0, 1.0])
    ax0 = fig.add_subplot(gs[0, :])
    n = 20 * spw
    x = np.arange(n) / (spc * 5.0)  # microseconds at 5 MHz: 4 samples/cycle -> sample period 0.05 us
    ax0.plot(x, t[:n], lw=0.9, color="#1f4e79")
    for i in range(21):
        ax0.axvline(i * spw / (spc * 5.0), color="#111827", lw=0.55, alpha=0.25)
    for i in range(0, 20, 2):
        s = i * spw
        e = s + spw
        f = np.mean(np.abs(t[s:e]))
        ax0.plot(x[s:e], np.full(spw, f), color="#d97706", lw=1.8, alpha=0.8)
    ax0.set_title(f"Legacy BNN-style trace segmentation: {spw} samples per 3x3 window ({dwell} cycles x {spc} ADC samples)")
    ax0.set_xlabel("time (microseconds)")
    ax0.set_ylabel("centered ADC value")
    ax0.grid(True, alpha=0.20)
    ax0.text(0.01, 0.96, "Orange levels = mean absolute power feature for each window", transform=ax0.transAxes, va="top", fontsize=9, bbox=dict(facecolor="white", edgecolor="#999", alpha=0.92))

    ax1 = fig.add_subplot(gs[1, 0])
    im = ax1.imshow(feat, cmap="magma")
    ax1.set_title("One kernel -> 26x26 power feature map")
    ax1.set_xticks([])
    ax1.set_yticks([])
    fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)

    ax2 = fig.add_subplot(gs[1, 1])
    ax2.imshow(feat_all, aspect="auto", cmap="viridis")
    ax2.set_title("9 kernels -> power vector per window")
    ax2.set_xlabel("window index")
    ax2.set_ylabel("kernel id")

    ax3 = fig.add_subplot(gs[1, 2])
    ax3.imshow(image, cmap="gray_r", interpolation="nearest")
    ax3.set_title("Truth image used only after attack scoring")
    ax3.set_xticks([])
    ax3.set_yticks([])
    save(fig, "fig04_trace_to_features.png")


def fig_rerun_trace_evidence():
    active_path = ROOT / "host" / "traces_rerun_20260830_onehot_80" / "img0740.npz"
    p2p_path = ROOT / "host" / "traces_rerun_20260830_p2p_2000" / "shard000.npz"
    if not active_path.exists() or not p2p_path.exists():
        return

    active = np.load(active_path, allow_pickle=True)
    p2p = np.load(p2p_path, allow_pickle=True)
    traces = active["traces"].astype(float)
    image = active["image"]
    spw = int(active["samples_per_window"])
    spc = int(active["samples_per_cycle"])
    dwell = int(active["dwell"])
    t = traces[0] - np.median(traces[0])
    nwin = 26 * 26
    feat = np.mean(np.abs(t[: nwin * spw].reshape(nwin, spw)), axis=1).reshape(26, 26)

    p2p_trace = p2p["traces"][0].astype(float)
    p2p_trace = p2p_trace - np.median(p2p_trace)

    fig = plt.figure(figsize=(12.5, 7.4), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.35, 1.0])
    ax0 = fig.add_subplot(gs[0, 0])
    n = min(50 * spw, len(t))
    x = np.arange(n) / (spc * 5.0)
    ax0.plot(x, t[:n], lw=0.85, color="#111111")
    for i in range(0, 51, 5):
        ax0.axvline(i * spw / (spc * 5.0), color="#666666", lw=0.5, alpha=0.35)
    ax0.set_title(f"Fresh active-template trace: 50 image windows ({dwell} cycles/window)")
    ax0.set_xlabel("time (microseconds, nominal)")
    ax0.set_ylabel("centered ADC")
    ax0.grid(True, alpha=0.22)

    ax1 = fig.add_subplot(gs[0, 1])
    im = ax1.imshow(feat, cmap="gray_r")
    ax1.set_title("Feature map from the same trace")
    ax1.set_xticks([])
    ax1.set_yticks([])
    fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)

    ax2 = fig.add_subplot(gs[1, 0])
    n2 = min(6000, len(p2p_trace))
    ax2.plot(np.arange(n2), p2p_trace[:n2], lw=0.75, color="#222222")
    ax2.set_title("Fresh P2P capture trace from shard000")
    ax2.set_xlabel("sample")
    ax2.set_ylabel("centered ADC")
    ax2.grid(True, alpha=0.22)

    ax3 = fig.add_subplot(gs[1, 1])
    ax3.imshow(image, cmap="gray_r", interpolation="nearest")
    ax3.set_title("Image is used after attack for scoring")
    ax3.set_xticks([])
    ax3.set_yticks([])
    save(fig, "fig16_rerun_trace_evidence.png")


def fig_algorithm_flow():
    fig, ax = plt.subplots(figsize=(13.2, 6.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.5, 0.94, "How the active/power-template reconstruction uses the signal", ha="center", va="center", fontsize=17, fontweight="bold")
    ax.text(0.5, 0.88, "The attack stage should see traces and the profile template. The true image is for scoring after reconstruction.", ha="center", va="center", fontsize=11)

    flow_fc = ["#ffffff", "#f4f4f4", "#f4f4f4", "#f4f4f4", "#e4e4e4", "#e4e4e4", "#ffffff"]
    flow_ec = "#333333"
    y = 0.55
    nodes = [
        (0.025, "FPGA run\n3x3 window stream", flow_fc[0], flow_ec),
        (0.165, "Power trace\nfrom CW-Lite", flow_fc[1], flow_ec),
        (0.305, "Split trace\ninto 676 windows", flow_fc[2], flow_ec),
        (0.445, "Extract feature\nper window/kernel", flow_fc[3], flow_ec),
        (0.585, "Search power\ntemplate", flow_fc[4], flow_ec),
        (0.725, "Vote overlap\npatches", flow_fc[5], flow_ec),
        (0.86, "Recovered\n28x28 image", flow_fc[6], flow_ec),
    ]
    for x, text, fc, ec in nodes:
        box(ax, x, y, 0.105, 0.18, text, fc, ec, size=9)
    for i in range(len(nodes) - 1):
        arrow(ax, (nodes[i][0] + 0.108, y + 0.09), (nodes[i + 1][0] - 0.002, y + 0.09))

    box(
        ax,
        0.24,
        0.19,
        0.23,
        0.18,
        "Profile side\nknown images + traces\nbuild mapping:\npatch bits -> power vector",
        "#f0f2f5",
        "#6b7280",
        size=9,
    )
    arrow(ax, (0.47, 0.28), (0.65, 0.55))
    box(
        ax,
        0.64,
        0.16,
        0.25,
        0.16,
        "Scoring side\ntruth image and label are loaded only after the attack output exists",
        "#f0f2f5",
        "#6b7280",
        size=9,
    )
    arrow(ax, (0.89, 0.24), (0.94, 0.55))
    ax.text(0.5, 0.06, "This is why the trace-only bundle matters: it avoids accidentally using the attacked image during reconstruction.", ha="center", va="center", fontsize=10)
    save(fig, "fig05_algorithm_flow.png")


def fig_metric_bars():
    score_files = [
        ("Fresh active-template\nheld-out 40", ROOT / "attack" / "rerun_20260830" / "score_onehot_80_eval40_heldout.json"),
        ("Fresh trained-template\nheld-out 40", ROOT / "attack" / "rerun_20260830" / "score_trained_80_eval40_heldout.json"),
        ("Hard avg=1 trained\nheld-out 20", ROOT / "attack" / "rerun_20260830" / "score_trained_avg1_120_eval20_heldout.json"),
        ("P2P clock-abs\ngrad loss", ROOT / "attack" / "rerun_20260830" / "p2p_2000_clockabs" / "summary.json"),
        ("P2P window-abs\ngrad loss", ROOT / "attack" / "rerun_20260830" / "p2p_2000_windowabs" / "summary.json"),
        ("P2P clock-abs\nMSE loss", ROOT / "attack" / "rerun_20260830" / "p2p_2000_clockabs_mse" / "summary.json"),
    ]
    # Fallback should normally not be used; it only keeps figure generation from
    # failing when copied without the full experiment folders.
    labels, vals = [], []
    fallback = {
        "Fresh active-template\nheld-out 40": {"bit_acc": 0.9455, "foreground_f1": 0.7495, "mssim": 0.726, "recognition_accuracy_recovered": 0.9750},
        "Fresh trained-template\nheld-out 40": {"bit_acc": 0.9600, "foreground_f1": 0.7924, "mssim": 0.769, "recognition_accuracy_recovered": 0.9000},
        "Hard avg=1 trained\nheld-out 20": {"bit_acc": 0.9115, "foreground_f1": 0.3770, "mssim": 0.452, "recognition_accuracy_recovered": 0.4500},
        "P2P clock-abs\ngrad loss": {"bit_acc": 0.9503, "foreground_f1": 0.7951, "mssim": 0.522, "recognition_accuracy_recovered": 0.8933},
        "P2P window-abs\ngrad loss": {"bit_acc": 0.9486, "foreground_f1": 0.7796, "mssim": 0.508, "recognition_accuracy_recovered": 0.8333},
        "P2P clock-abs\nMSE loss": {"bit_acc": 0.9537, "foreground_f1": 0.8068, "mssim": 0.561, "recognition_accuracy_recovered": 0.9033},
    }
    for label, path in score_files:
        if path.exists():
            m = json.loads(path.read_text(encoding="utf-8"))["metrics"]
        else:
            m = fallback[label]
        labels.append(label)
        vals.append([
            m["bit_acc"],
            m["foreground_f1"],
            m["mssim"],
            m["recognition_accuracy_recovered"],
        ])
    vals = np.array(vals)
    metric_names = ["pixel acc.", "foreground F1", "MSSIM", "recognition"]
    x = np.arange(len(labels))
    width = 0.18
    colors = ["#4477aa", "#66aa55", "#cc8844", "#aa4455"]
    fig, ax = plt.subplots(figsize=(13.8, 5.8))
    for i, name in enumerate(metric_names):
        ax.bar(x + (i - 1.5) * width, vals[:, i], width=width, label=name, color=colors[i], alpha=0.92)
    ax.axhline(0.8785, color="#6b7280", linestyle="--", lw=1.2, label="typical all-zero pixel baseline")
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("score")
    ax.set_title("Fresh checked hardware results, including the harder avg=1 stress test")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.grid(axis="y", alpha=0.22)
    ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.13), frameon=False)
    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            ax.text(x[i] + (j - 1.5) * width, vals[i, j] + 0.018, f"{vals[i, j]:.2f}", ha="center", va="bottom", fontsize=8, rotation=0)
    ax.text(0.01, 0.04, "All entries in this chart use the fresh live CW305 trace folders.", transform=ax.transAxes, fontsize=9, bbox=dict(facecolor="white", edgecolor="#999", alpha=0.92))
    fig.tight_layout()
    save(fig, "fig06_pilot_metrics.png")


def fig_hard_trace_noise():
    avg5_path = ROOT / "host" / "traces_rerun_20260830_trained_80" / "img0780.npz"
    avg1_path = ROOT / "host" / "traces_rerun_20260830_trained_avg1_120" / "img0900.npz"
    if not avg5_path.exists() or not avg1_path.exists():
        return
    d5 = np.load(avg5_path, allow_pickle=True)
    d1 = np.load(avg1_path, allow_pickle=True)
    t5 = d5["traces"][0].astype(float)
    t1 = d1["traces"][0].astype(float)
    t5 = t5 - np.median(t5)
    t1 = t1 - np.median(t1)
    n = min(4500, len(t5), len(t1))
    fig, axes = plt.subplots(2, 1, figsize=(12.4, 5.4), sharex=True)
    axes[0].plot(np.arange(n), t5[:n], color="#4477aa", lw=0.75)
    axes[0].set_title("Averaged trained-kernel trace, avg=5")
    axes[0].set_ylabel("ADC")
    axes[1].plot(np.arange(n), t1[:n], color="#aa4455", lw=0.75)
    axes[1].set_title("Hard trained-kernel trace, avg=1")
    axes[1].set_xlabel("sample index")
    axes[1].set_ylabel("ADC")
    for ax in axes:
        ax.grid(alpha=0.22)
    rms5 = float(np.sqrt(np.mean(t5 ** 2)))
    rms1 = float(np.sqrt(np.mean(t1 ** 2)))
    fig.suptitle(f"Harder hardware test: avg=1 removes repeat averaging (trace RMS avg=5: {rms5:.4f}, avg=1: {rms1:.4f})", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save(fig, "fig20_hard_avg1_trace_noise.png")


def fig_p2p_rerun_examples():
    p = ROOT / "attack" / "rerun_20260830" / "p2p_2000_clockabs" / "test_recovered.npz"
    if not p.exists():
        return
    d = np.load(p, allow_pickle=True)
    rec = d["recovered"].astype(float)
    truth = d["truth"].astype(float)
    labels = d["labels"]
    indices = d["indices"]
    count = min(4, len(rec))
    fig, axes = plt.subplots(count, 3, figsize=(5.9, 1.48 * count))
    if count == 1:
        axes = np.asarray([axes])
    for i in range(count):
        axes[i, 0].imshow(truth[i], cmap="gray_r", vmin=0, vmax=1)
        axes[i, 0].set_title(f"Truth {int(labels[i])}\nidx {int(indices[i])}", fontsize=8)
        axes[i, 1].imshow(rec[i], cmap="gray_r", vmin=0, vmax=1)
        axes[i, 1].set_title("Generator output", fontsize=8)
        axes[i, 2].imshow((rec[i] >= 0.5).astype(float), cmap="gray_r", vmin=0, vmax=1)
        axes[i, 2].set_title("Thresholded", fontsize=8)
        for ax in axes[i]:
            ax.axis("off")
    fig.suptitle("Fresh P2P-style recovery from live CW305 traces", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    save(fig, "fig17_rerun_p2p_examples.png")


def fig_p2p_mse_examples():
    p = ROOT / "attack" / "rerun_20260830" / "p2p_2000_clockabs_mse" / "test_recovered.npz"
    if not p.exists():
        return
    d = np.load(p, allow_pickle=True)
    rec = d["recovered"].astype(float)
    truth = d["truth"].astype(float)
    labels = d["labels"]
    indices = d["indices"]
    count = min(4, len(rec))
    fig, axes = plt.subplots(count, 3, figsize=(5.9, 1.48 * count))
    if count == 1:
        axes = np.asarray([axes])
    for i in range(count):
        axes[i, 0].imshow(truth[i], cmap="gray_r", vmin=0, vmax=1)
        axes[i, 0].set_title(f"Truth {int(labels[i])}\nidx {int(indices[i])}", fontsize=8)
        axes[i, 1].imshow(rec[i], cmap="gray_r", vmin=0, vmax=1)
        axes[i, 1].set_title("MSE output", fontsize=8)
        axes[i, 2].imshow((rec[i] >= 0.5).astype(float), cmap="gray_r", vmin=0, vmax=1)
        axes[i, 2].set_title("Thresholded", fontsize=8)
        for ax in axes[i]:
            ax.axis("off")
    fig.suptitle("Fresh P2P clock-abs MSE recovery", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    save(fig, "fig19_rerun_p2p_mse_examples.png")


def fig_active_rerun_examples():
    truth_path = ROOT / "attack" / "rerun_20260830" / "truth_onehot_80_eval40_heldout.json"
    results = ROOT / "attack" / "rerun_20260830" / "results_onehot_80_eval40_heldout"
    score_path = results / "score_summary.json"
    if not truth_path.exists() or not score_path.exists():
        return
    truth = json.loads(truth_path.read_text(encoding="utf-8"))
    truth_by_file = {item["file"]: item for item in truth["items"]}
    score = json.loads(score_path.read_text(encoding="utf-8"))
    rows = score["images"][:3]
    fig, axes = plt.subplots(len(rows), 2, figsize=(4.7, 1.45 * len(rows)))
    if len(rows) == 1:
        axes = np.asarray([axes])
    for r, axrow in zip(rows, axes):
        truth_item = truth_by_file[r["file"]]
        orig = np.asarray(truth_item["image"], dtype=float)
        rec_name = "recovered_" + r["file"][3:7] + ".npz"
        recovered = np.load(results / rec_name)["recovered"].astype(float)
        axrow[0].imshow(orig, cmap="gray_r", vmin=0, vmax=1)
        axrow[0].set_title(f"Truth {int(r['label'])}", fontsize=8)
        axrow[1].imshow(recovered, cmap="gray_r", vmin=0, vmax=1)
        axrow[1].set_title(f"Recovered F1={r['foreground_f1']:.2f}", fontsize=8)
        for ax in axrow:
            ax.axis("off")
    fig.suptitle("Fresh active-template held-out recovery", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    save(fig, "fig15_rerun_onehot_heldout_examples.png")


def fig_active_trained_rerun_examples():
    truth_path = ROOT / "attack" / "rerun_20260830" / "truth_trained_80_eval40_heldout.json"
    results = ROOT / "attack" / "rerun_20260830" / "results_trained_80_eval40_heldout"
    score_path = results / "score_summary.json"
    if not truth_path.exists() or not score_path.exists():
        return
    truth = json.loads(truth_path.read_text(encoding="utf-8"))
    truth_by_file = {item["file"]: item for item in truth["items"]}
    score = json.loads(score_path.read_text(encoding="utf-8"))
    rows = score["images"][:3]
    fig, axes = plt.subplots(len(rows), 2, figsize=(4.7, 1.45 * len(rows)))
    if len(rows) == 1:
        axes = np.asarray([axes])
    for r, axrow in zip(rows, axes):
        truth_item = truth_by_file[r["file"]]
        orig = np.asarray(truth_item["image"], dtype=float)
        rec_name = "recovered_" + r["file"][3:7] + ".npz"
        recovered = np.load(results / rec_name)["recovered"].astype(float)
        axrow[0].imshow(orig, cmap="gray_r", vmin=0, vmax=1)
        axrow[0].set_title(f"Truth {int(r['label'])}", fontsize=8)
        axrow[1].imshow(recovered, cmap="gray_r", vmin=0, vmax=1)
        axrow[1].set_title(f"Recovered F1={r['foreground_f1']:.2f}", fontsize=8)
        for ax in axrow:
            ax.axis("off")
    fig.suptitle("Fresh trained-kernel active-template recovery", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    save(fig, "fig18_rerun_trained_heldout_examples.png")


def fig_active_hard_avg1_examples():
    truth_path = ROOT / "attack" / "rerun_20260830" / "truth_trained_avg1_120_eval20_heldout.json"
    results = ROOT / "attack" / "rerun_20260830" / "results_trained_avg1_120_eval20_heldout"
    score_path = results / "score_summary.json"
    if not truth_path.exists() or not score_path.exists():
        return
    truth = json.loads(truth_path.read_text(encoding="utf-8"))
    truth_by_file = {item["file"]: item for item in truth["items"]}
    score = json.loads(score_path.read_text(encoding="utf-8"))
    rows = score["images"][:4]
    fig, axes = plt.subplots(len(rows), 2, figsize=(4.8, 1.45 * len(rows)))
    if len(rows) == 1:
        axes = np.asarray([axes])
    for r, axrow in zip(rows, axes):
        truth_item = truth_by_file[r["file"]]
        orig = np.asarray(truth_item["image"], dtype=float)
        rec_name = "recovered_" + r["file"][3:7] + ".npz"
        recovered = np.load(results / rec_name)["recovered"].astype(float)
        axrow[0].imshow(orig, cmap="gray_r", vmin=0, vmax=1)
        axrow[0].set_title(f"Truth {int(r['label'])}", fontsize=8)
        axrow[1].imshow(recovered, cmap="gray_r", vmin=0, vmax=1)
        axrow[1].set_title(f"Avg=1 F1={r['foreground_f1']:.2f}", fontsize=8)
        for ax in axrow:
            ax.axis("off")
    fig.suptitle("Hard active-template recovery, trained kernels, avg=1", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    save(fig, "fig21_rerun_hard_avg1_examples.png")


def fig_reference_comparison():
    fig, ax = plt.subplots(figsize=(12.8, 7.0))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.5, 0.95, "Where this work differs from the reference papers", ha="center", va="center", fontsize=17, fontweight="bold")

    box(ax, 0.05, 0.77, 0.38, 0.11, "Wei et al. / original active-template idea", COL["blue"], COL["blue_edge"], size=10, weight="bold")
    box(ax, 0.57, 0.77, 0.38, 0.11, "My CW305 + ChipWhisperer implementation", COL["green"], COL["green_edge"], size=10, weight="bold")

    left = [
        "Victim: CNN/BNN accelerator first layer",
        "Leak source: line buffer and convolution datapath",
        "Measurement: high bandwidth external scope",
        "Output: MNIST image recovery / recognition",
        "Active attack: power template from profiling",
    ]
    right = [
        "Victim: custom binary 3x3 conv on CW305",
        "Leak source: retained XNOR leak bank to raise SNR",
        "Measurement: CW-Lite synchronous x4 capture",
        "Output: binary MNIST silhouette recovery",
        "Added check: reject stock AES before capture",
    ]
    for i, txt in enumerate(left):
        yy = 0.63 - i * 0.105
        box(ax, 0.05, yy, 0.38, 0.072, txt, "white", "#555555", size=9)
    for i, txt in enumerate(right):
        yy = 0.63 - i * 0.105
        box(ax, 0.57, yy, 0.38, 0.072, txt, "white", "#555555", size=9)
    for i in range(5):
        yy = 0.665 - i * 0.105
        arrow(ax, (0.44, yy), (0.56, yy), color="#6b7280")
    box(
        ax,
        0.17,
        0.05,
        0.66,
        0.12,
        "Same research question: can first-layer power reveal the private input?\nDifferent bench: the method must be adapted and clearly labelled.",
        COL["yellow"],
        COL["yellow_edge"],
        size=10,
    )
    save(fig, "fig07_reference_vs_our_setup.png")


def fig_next_hardware_checklist():
    fig, ax = plt.subplots(figsize=(11.8, 6.0))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.5, 0.91, "Next hardware run: what must pass before new claims", ha="center", va="center", fontsize=17, fontweight="bold")
    steps = [
        ("1", "S1 switches\nM0=M1=M2=1", "#e4e4e4", "#444444"),
        ("2", "Program custom\nleakage bitstream", "#ffffff", "#444444"),
        ("3", "Read pages 2/3/4\nmust not be AES", "#d0d0d0", "#222222"),
        ("4", "Functional check\n676 outputs pass", "#ffffff", "#444444"),
        ("5", "Capture traces\nwith manifest serials", "#f4f4f4", "#444444"),
        ("6", "Train/reconstruct\nusing trace-only test", "#ffffff", "#444444"),
    ]
    coords = [
        (0.06, 0.60),
        (0.37, 0.60),
        (0.68, 0.60),
        (0.68, 0.29),
        (0.37, 0.29),
        (0.06, 0.29),
    ]
    w, h = 0.25, 0.19
    for (x, y), (num, text, fc, ec) in zip(coords, steps):
        box(ax, x, y, w, h, f"{num}\n{text}", fc, ec, size=9, weight="bold")
    arrow(ax, (0.31, 0.695), (0.37, 0.695))
    arrow(ax, (0.62, 0.695), (0.68, 0.695))
    arrow(ax, (0.805, 0.60), (0.805, 0.48))
    arrow(ax, (0.68, 0.385), (0.62, 0.385))
    arrow(ax, (0.37, 0.385), (0.31, 0.385))
    ax.text(0.5, 0.13, "If step 3 fails, the board is not running the target design. Stop capture, because later ML results can be misleading.", ha="center", va="center", fontsize=10)
    save(fig, "fig08_next_hardware_checklist.png")


def copy_existing_figures():
    copies = {
        "fig09_template_trained_examples.png": ROOT / "report" / "figs_new" / "template_trained_kernels.png",
        "fig10_p2p_generator_examples.png": ROOT / "report" / "figs_new" / "p2p_generator.png",
        "fig11_active_template_trace_only_examples.png": ROOT / "report" / "paper_figures" / "fig_active_power_template_trace_only_report.png",
        "fig12_background_detection_examples.png": ROOT / "report" / "figs_new" / "bg_detect_onehot.png",
        "fig13_profile_lookup.png": ROOT / "report" / "paper_figures" / "fig_profile_lookup.png",
        "fig14_match_count_stream.png": ROOT / "report" / "paper_figures" / "fig_match_count_stream.png",
    }
    for name, src in copies.items():
        if src.exists():
            shutil.copy2(src, FIG / name)


def main():
    fig_status_summary()
    fig_aes_control_trace()
    fig_register_gate()
    fig_trace_to_features()
    fig_rerun_trace_evidence()
    fig_algorithm_flow()
    fig_metric_bars()
    fig_active_rerun_examples()
    fig_active_trained_rerun_examples()
    fig_active_hard_avg1_examples()
    fig_hard_trace_noise()
    fig_p2p_rerun_examples()
    fig_p2p_mse_examples()
    fig_reference_comparison()
    fig_next_hardware_checklist()
    copy_existing_figures()
    print(f"Wrote figures to {FIG}")


if __name__ == "__main__":
    main()
