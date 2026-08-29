"""
P3 -- S6 Background Detection (passive adversary).

Intuition (paper S6.1): cycles whose convolution window pixels are all equal (pure
background) cause almost no datapath switching -> low per-cycle power. Threshold the
per-cycle power; low-power cycles are background pixels. Recovers a black/white
silhouette. Works from ONE kernel's trace (paper: kernel choice barely matters).
"""
import numpy as np
from common import LINE, NCYC, cyc_to_yx


def choose_threshold(power, method="eq4", nbins=100):
    """Threshold separating background (low-power) from foreground (high-power) cycles.

    method="eq4" (paper-faithful, Section 6.2 Eq.4):
        PT = argmax_P [ C(P - B) - C(P) ]  -- the power value at the sharpest DROP in the
        per-cycle-power histogram (C = cycle count, B = bin size). The paper observes the
        count rise to a background peak then descend sharply; the max decrease marks the
        background/foreground boundary.
    method="otsu": bimodal inter-class-variance split (more robust when a tall near-zero
        spike makes Eq.4 latch too low, as for 5x5).
    """
    pw = np.asarray(power, float)
    if method == "otsu":
        c, e = np.histogram(pw, bins=max(nbins, 128))
        c = c.astype(float); p = c / max(c.sum(), 1e-12)
        mids = 0.5 * (e[:-1] + e[1:])
        w0 = np.cumsum(p); w1 = 1.0 - w0
        m0 = np.cumsum(p * mids) / np.clip(w0, 1e-12, None)
        mt = (p * mids).sum()
        m1 = (mt - np.cumsum(p * mids)) / np.clip(w1, 1e-12, None)
        sigma = w0 * w1 * (m0 - m1) ** 2
        return float(mids[int(np.nanargmax(sigma))])
    # paper Eq.4: sharpest decrease in cycle count
    counts, edges = np.histogram(pw, bins=nbins)
    drop = counts[:-1].astype(int) - counts[1:].astype(int)   # C(P) - C(P+B)
    k = int(np.argmax(drop))
    return float(edges[k + 1])


def recover_background(power_one_kernel, thr=None, method="otsu"):
    """
    power_one_kernel: [784] per-cycle power for a single kernel.
    Returns (binary_img 28x28 uint8 {0=background,255=foreground}, marker 28x28 {0/1},
             threshold). marker[y,x]=1 means predicted FOREGROUND.

    Default threshold is Otsu, not the paper's Eq.4. With the FANOUT-amplified leakage the
    per-cycle-power histogram has a tall near-zero background spike that makes Eq.4's
    "sharpest count drop" latch too low (~0.78 pixel acc); Otsu's bimodal split lands near
    the optimum (~0.93, above the paper's 0.862). Pass method="eq4" for the paper-literal
    rule or thr=<value> to override.
    """
    pw = np.asarray(power_one_kernel, float)
    if thr is None:
        thr = choose_threshold(pw, method=method)
    marker = np.zeros((LINE, LINE), dtype=np.uint8)
    for c in range(NCYC):
        y, x = cyc_to_yx(c)
        marker[y, x] = 1 if pw[c] > thr else 0   # above thr -> foreground
    img = (marker * 255).astype(np.uint8)
    return img, marker, thr


def pixel_accuracy(marker_pred, image):
    """Paper Eq.5: fraction of pixels whose background/foreground marker is correct.
    True background = pixel value 0 (pure black)."""
    true_fg = (image > 0).astype(np.uint8)
    return float((marker_pred == true_fg).mean())
