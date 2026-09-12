"""Predefined task transforms. Fixed BEFORE calibration; never chosen by results.

A pooled RGB score answers one blunt question. These decompose it, because the
repository's own evidence already separates the answers: structure improves while
chrominance does not, and a single number cannot say that.

PRIMARY tasks are full-resolution Y, Cb and Cr. Cb and Cr are kept SEPARATE -- the
existing bootstrap averaged them, which can hide one channel improving while the other
degrades.

SECONDARY is an 8x8 area-averaged Y. It is a coarse-structure diagnostic and is never a
substitute for full-resolution fidelity: a method can score well on block means while
being wrong at every pixel.
"""
from __future__ import annotations

import numpy as np

# ITU-R BT.601 on the stored encoding. This is a numerical transform of whatever the
# capture pipeline wrote; it does not assert a radiometrically linear colour space.
_KR, _KG, _KB = 0.299, 0.587, 0.114


def _ycbcr(x):
    r, g, b = x[:, 0], x[:, 1], x[:, 2]
    y = _KR * r + _KG * g + _KB * b
    return y, 0.564 * (b - y), 0.713 * (r - y)


def task_Y(x):
    return _ycbcr(np.clip(x, 0, 1))[0]


def task_Cb(x):
    return _ycbcr(np.clip(x, 0, 1))[1]


def task_Cr(x):
    return _ycbcr(np.clip(x, 0, 1))[2]


def task_Y_coarse(x, block=4):
    """8x8 area-averaged luminance for 32x32 inputs. SECONDARY diagnostic only."""
    y = task_Y(x)
    n, h, w = y.shape
    if h % block or w % block:
        raise ValueError(f"{h}x{w} not divisible by block {block}")
    return y.reshape(n, h // block, block, w // block, block).mean(axis=(2, 4))


PRIMARY = {"Y": task_Y, "Cb": task_Cb, "Cr": task_Cr}
SECONDARY = {"Y_coarse8": task_Y_coarse}
ALL = {**PRIMARY, **SECONDARY}


def task_scores(pred: np.ndarray, truth: np.ndarray, name: str) -> np.ndarray:
    """Per-example task error, e_{i,t} = (255/d_t) * ||h_t(pred) - h_t(truth)||_1.

    Returns one scalar per example, in the repository's 0-255 error units so the numbers
    are comparable to every MAE reported elsewhere.
    """
    h = ALL[name]
    a, b = h(pred), h(truth)
    n = len(a)
    d_t = int(np.prod(a.shape[1:]))
    return (255.0 / d_t) * np.abs(a.reshape(n, -1) - b.reshape(n, -1)).sum(axis=1)
