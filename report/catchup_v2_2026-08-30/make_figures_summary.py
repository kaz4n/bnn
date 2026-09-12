#!/usr/bin/env python3
"""Figure variants used only by the summary report.

The summary drops the calendar dates, so the authenticity-gate figure is redrawn
with the two columns labelled by what they show rather than by the day they were
captured. Everything else, including the original fig02, is left untouched.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from make_figures import (DEEP, FILL_SOFT, FILL_STRONG, MUTE, arrow, blank,
                          box, save)


def fig_authenticity_nodates():
    fig, ax = plt.subplots(figsize=(11.6, 4.6))
    blank(ax)
    ax.set_title("The gate that refused the bad session and passed the good one. "
                 "This is why I trust these traces.", loc="left",
                 fontsize=11.5, weight="bold", pad=12)

    ax.text(0.24, 0.93, "Before the fix", ha="center", fontsize=11, weight="bold", color=MUTE)
    ax.text(0.76, 0.93, "After the fix", ha="center", fontsize=11, weight="bold", color=DEEP)
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
    save(fig, "fig02_authenticity_gate_nodates.png")


if __name__ == "__main__":
    fig_authenticity_nodates()
