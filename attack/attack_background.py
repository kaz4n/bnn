"""
P3 -- S6 Background Detection (passive adversary).

Intuition (paper S6.1): cycles whose convolution window pixels are all equal (pure
background) cause almost no datapath switching -> low per-cycle power. Threshold the
per-cycle power; low-power cycles are background pixels. Recovers a black/white
silhouette. Works from ONE kernel's trace (paper: kernel choice barely matters).
"""
import numpy as np
from common import LINE, NCYC, cyc_to_yx


def choose_threshold(power, nbins=60):
    """Paper Eq.4: PT = argmax_P [ C(P - B) - C(P) ]  (max decrease in cycle count)."""
    counts, edges = np.histogram(power, bins=nbins)
    diff = counts[:-1] - counts[1:]              # C(P) - C(P+B); peak drop after rise
    k = int(np.argmax(diff))
    return 0.5 * (edges[k + 1] + edges[k + 2])


def recover_background(power_one_kernel, thr=None):
    """
    power_one_kernel: [784] per-cycle power for a single kernel.
    Returns (binary_img 28x28 uint8 {0=background,255=foreground}, marker 28x28 {0/1},
             threshold). marker[y,x]=1 means predicted FOREGROUND.
    """
    pw = np.asarray(power_one_kernel, float)
    if thr is None:
        thr = choose_threshold(pw)
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
