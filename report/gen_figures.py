#!/usr/bin/env python3
"""Generate figures for the LaTeX report from real captured + simulated data."""
import os, sys, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.join(os.path.dirname(__file__), "..")
FIG = os.path.join(os.path.dirname(__file__), "figs")
os.makedirs(FIG, exist_ok=True)
EW, NCONV, K, SPC = 30, 900, 3, 4


def winmag(img):
    pad = np.zeros((EW, EW), int); pad[1:29, 1:29] = img
    m = np.zeros(NCONV)
    for c in range(NCONV):
        er, ec = c // EW, c % EW
        for i in range(K):
            for j in range(K):
                yy, xx = er-2+i, ec-2+j
                if 0 <= yy < EW and 0 <= xx < EW:
                    m[c] += pad[yy, xx]
    return m


# 1) P0 AES trace (bench works)
w = np.load(os.path.join(ROOT, "p0_trace.npy"))
plt.figure(figsize=(7, 2.2)); plt.plot(w, lw=0.4)
plt.title("(P0) Stock AES power trace on CW305 -- bench validated")
plt.xlabel("sample (synchronous, 4x clock)"); plt.ylabel("power (V)")
plt.tight_layout(); plt.savefig(f"{FIG}/p0_aes.png", dpi=120); plt.close()

# 2) Real BNN conv trace (one kernel, averaged)
d = np.load(os.path.join(ROOT, "host/traces_hi/img0000.npz"))
tr = d["traces"][0][:NCONV*SPC]
plt.figure(figsize=(7, 2.2)); plt.plot(tr, lw=0.4, color="C3")
plt.title("(P3) Layer-1 BNN conv power trace, CW305+CW-Lite (kernel 0, avg=100)")
plt.xlabel("sample (4 / clock cycle, 900 cycles)"); plt.ylabel("power (V, AC-coupled)")
plt.tight_layout(); plt.savefig(f"{FIG}/real_trace.png", dpi=120); plt.close()

# 3) per-cycle power vs window magnitude (they don't track) + scatter
pc = np.abs(d["traces"][0][:NCONV*SPC].reshape(NCONV, SPC)).sum(1)
mag = winmag(d["image"].astype(int))
n = 140
fig, ax = plt.subplots(2, 1, figsize=(7, 3.6))
a = ax[0]
a.plot((pc[:n]-pc[:n].mean())/pc[:n].std(), lw=0.8, label="measured per-cycle power")
a.plot((mag[:n]-mag[:n].mean())/mag[:n].std(), lw=0.8, label="true window pixel magnitude")
a.set_title("Per-cycle power vs. true pixel content (first 140 cycles) -- no tracking")
a.set_xlabel("conv cycle"); a.legend(fontsize=7); a.set_ylabel("z-score")
r = np.corrcoef(pc, mag)[0, 1]
ax[1].scatter(mag, pc, s=3, alpha=0.3)
ax[1].set_title(f"Scatter: corr = {abs(r):.3f}  (recovery needs >~0.5)")
ax[1].set_xlabel("window pixel magnitude"); ax[1].set_ylabel("per-cycle power")
plt.tight_layout(); plt.savefig(f"{FIG}/corr_fail.png", dpi=120); plt.close()

# 4) Simulation recovery grid (paper-level) -- original / background / template
files = sorted(glob.glob(os.path.join(ROOT, "attack/results_3x3/recovered_*.npz")))[:6]
fig, ax = plt.subplots(3, len(files), figsize=(1.5*len(files), 4.6))
rows = ["original", "background", "template"]
for c, f in enumerate(files):
    z = np.load(f)
    for r_, key in enumerate(rows):
        ax[r_, c].imshow(z[key], cmap="gray"); ax[r_, c].axis("off")
        if c == 0: ax[r_, c].set_ylabel(key, rotation=90)
for r_, key in enumerate(rows):
    ax[r_, 0].set_title("") ; ax[r_, 0].text(-8, 14, key, rotation=90, va="center", fontsize=9)
fig.suptitle("Simulation: input recovered from modelled power (3x3) -- template recog 0.935 (paper 0.898)")
plt.tight_layout(); plt.savefig(f"{FIG}/sim_recovery.png", dpi=120); plt.close()

# 5) correlation across design variants (the wall)
variants = ["v1 preload\n+mux", "v2 addressed\nline buf", "v3 shift-reg\nline buf",
            "v4 16x\namplifier", "async\n21x/cyc"]
corr = [0.10, 0.14, 0.13, 0.10, 0.13]
plt.figure(figsize=(6, 2.6))
plt.bar(range(len(variants)), corr, color="C3")
plt.axhline(0.5, ls="--", color="green", label="recovery threshold (~0.5)")
plt.xticks(range(len(variants)), variants, fontsize=7)
plt.ylabel("corr(power, pixels)"); plt.ylim(0, 0.6)
plt.title("Every design lever hits the same ~0.1 measurement floor")
plt.legend(fontsize=8); plt.tight_layout(); plt.savefig(f"{FIG}/corr_wall.png", dpi=120); plt.close()

print("figures written to", FIG)
print(os.listdir(FIG))
