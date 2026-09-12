"""Split-conformal calibration on predefined task scores.

Given a FROZEN predictor and calibration scores exchangeable with a new example's, the
kth order statistic gives a radius with marginal coverage >= 1-alpha:

    k   = ceil((n+1)(1-alpha))
    q_t = e_(k),t          (kth smallest calibration score)

WHAT THIS IS NOT, stated here because the failure mode is interpretive rather than
numerical:

  * Not a posterior, and not the set of images physically consistent with a measurement.
  * MARGINAL, not conditional: it bounds the average task error of a new example at the
    stated rate over the population. It says nothing about any particular image, and
    nothing about a subgroup chosen after the fact.
  * Not joint across tasks. Three separate 90% statements are not a 90% joint statement.
  * Worthless without a usefulness check. A radius of 255 covers everything and means
    nothing, which is why every radius is reported beside the no-input baseline's.

When k > n the finite sample cannot support the requested level and the radius is
infinite. That is reported as infinite -- substituting the maximum observed score would
manufacture a bound the data does not support.
"""
from __future__ import annotations

import math

import numpy as np


def conformal_radius(cal_scores: np.ndarray, alpha: float = 0.10) -> dict:
    s = np.sort(np.asarray(cal_scores, dtype=np.float64))
    n = len(s)
    if n == 0:
        raise ValueError("no calibration scores")
    k = math.ceil((n + 1) * (1 - alpha))
    if k > n:
        return {"n_cal": n, "alpha": alpha, "k": k, "radius": float("inf"),
                "finite": False,
                "note": (f"k={k} exceeds n={n}: {n} calibration examples cannot support "
                         f"a {1-alpha:.0%} bound. Radius is infinite by construction; "
                         "the maximum observed score is NOT a valid substitute.")}
    return {"n_cal": n, "alpha": alpha, "k": k, "radius": float(s[k - 1]),
            "finite": True,
            "note": f"kth smallest of {n} calibration scores, k=ceil((n+1)(1-alpha))"}


def evaluate_coverage(radius: float, assess_scores: np.ndarray,
                      alpha: float = 0.10) -> dict:
    a = np.asarray(assess_scores, dtype=np.float64)
    n = len(a)
    covered = int(np.sum(a <= radius))
    rate = covered / n if n else float("nan")
    # Wilson interval: better behaved than normal-approximation at rates near 1, which
    # is exactly where coverage lives.
    z = 1.959963985
    if n:
        p = rate
        den = 1 + z * z / n
        centre = (p + z * z / (2 * n)) / den
        half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
        lo, hi = max(0.0, centre - half), min(1.0, centre + half)
    else:
        lo = hi = float("nan")
    return {"n_assess": n, "covered": covered, "coverage": rate,
            "wilson_95": [lo, hi], "target": 1 - alpha,
            "materially_undercovers": bool(hi < 1 - alpha),
            "note": ("Wilson interval treats assessment examples as independent. That "
                     "holds here only because each is a distinct source scene; with "
                     "repeated views of one scene it would be too narrow.")}
