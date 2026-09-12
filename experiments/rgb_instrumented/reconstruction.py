"""Demosaic intentionally exported, normalized RGGB Bayer measurements.

No acquisition, trace processing, or measurement inference is performed here.
The input is a floating NumPy array of shape (H, W), with H and W at least 2.
R is sampled at even/even coordinates, B at odd/odd, and G elsewhere. Output
is float64 RGB in [0, 1]; every measured value is retained exactly.

Methods and keyword defaults:

* ``nearest``: Euclidean nearest same-color sample; no keyword parameters.
  Equal-distance ties follow SciPy's distance-transform implementation.
* ``bilinear``: standard Bayer bilinear kernels, normalized over available
  samples at image boundaries; no keyword parameters.
* ``smooth_ridge``: minimize, independently in each channel,
  ``0.5 * sum_edges (u_i-u_j)**2 + ridge/2 * sum_i (u_i-b_i)**2``
  subject to exact measured samples. Edges join horizontal/vertical neighbors,
  and ``b`` is the bilinear image. This is graph-Laplacian inpainting with a
  ridge toward a specified prior, not ridge regression on a measurement model.
  Defaults: ``ridge=0.05, rtol=1e-7, atol=1e-10, maxiter=1000``.
* ``tv``: minimize ``0.5 * ||u-b||**2 + weight * TV(u)`` subject to exact
  measured samples and ``0 <= u <= 1``. TV sums isotropic spatial gradient
  norms separately for each RGB channel, with zero forward differences at
  the lower/right boundaries. A Chambolle--Pock primal-dual solver uses
  ``tau=sigma=0.35`` and extrapolation 1. Defaults:
  ``weight=0.05, rtol=1e-5, atol=1e-8, maxiter=2000, check_every=10``.
  It stops when the primal-dual gap is at most
  ``atol + rtol * max(abs(primal_objective), abs(dual_objective))``.

Both iterative methods raise :class:`ConvergenceError` on solver failure.
Their priors can smooth genuine image detail; no reconstruction-quality claim
is implied. Odd dimensions are supported, but either dimension below 2 is
rejected because some color channels then have no measured samples.

References:
https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.cg.html
https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.distance_transform_edt.html
https://www.cmap.polytechnique.fr/preprint/repository/685.pdf
"""

from __future__ import annotations

import inspect
from numbers import Integral, Real

import numpy as np
from scipy import ndimage, sparse
from scipy.sparse.linalg import cg

METHODS = ("nearest", "bilinear", "smooth_ridge", "tv")


class ConvergenceError(RuntimeError):
    """An iterative reconstruction did not satisfy its convergence criterion."""


def get_methods() -> tuple[str, ...]:
    """Return supported method identifiers in a stable order."""
    return METHODS


def _validate_mosaic(mosaic: np.ndarray) -> np.ndarray:
    if not isinstance(mosaic, np.ndarray):
        raise TypeError("mosaic must be a floating NumPy array")
    if not np.issubdtype(mosaic.dtype, np.floating):
        raise TypeError("mosaic must have a real floating dtype")
    if mosaic.ndim != 2 or min(mosaic.shape) < 2:
        raise ValueError("mosaic must have shape (H, W) with both dimensions >= 2")
    if not np.all(np.isfinite(mosaic)):
        raise ValueError("mosaic values must be finite")
    if np.any(mosaic < 0) or np.any(mosaic > 1):
        raise ValueError("mosaic values must lie in [0, 1]")
    return np.array(mosaic, dtype=np.float64, copy=True)


def _masks(shape: tuple[int, int]) -> np.ndarray:
    masks = np.zeros((*shape, 3), dtype=bool)
    masks[0::2, 0::2, 0] = True
    masks[1::2, 1::2, 2] = True
    masks[..., 1] = ~(masks[..., 0] | masks[..., 2])
    return masks


def _real_parameter(name: str, value: float, *, zero_allowed: bool) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    if not np.isfinite(value) or value < 0 or (not zero_allowed and value == 0):
        relation = "nonnegative" if zero_allowed else "positive"
        raise ValueError(f"{name} must be finite and {relation}")
    return float(value)


def _positive_integer(name: str, value: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be a positive integer")
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _nearest(mosaic: np.ndarray, masks: np.ndarray) -> np.ndarray:
    output = np.empty((*mosaic.shape, 3), dtype=np.float64)
    for channel in range(3):
        indices = ndimage.distance_transform_edt(
            ~masks[..., channel], return_distances=False, return_indices=True
        )
        output[..., channel] = mosaic[tuple(indices)]
    return output


def _bilinear(mosaic: np.ndarray, masks: np.ndarray) -> np.ndarray:
    rb_kernel = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=float)
    g_kernel = np.array([[0, 1, 0], [1, 4, 1], [0, 1, 0]], dtype=float)
    output = np.empty((*mosaic.shape, 3), dtype=np.float64)
    for channel, kernel in enumerate((rb_kernel, g_kernel, rb_kernel)):
        mask = masks[..., channel].astype(float)
        numerator = ndimage.convolve(mosaic * mask, kernel, mode="constant")
        denominator = ndimage.convolve(mask, kernel, mode="constant")
        output[..., channel] = numerator / denominator
    return output


def _grid_laplacian(shape: tuple[int, int]) -> sparse.csr_matrix:
    height, width = shape
    size = height * width
    horizontal = np.ones(size - 1)
    horizontal[np.arange(1, height) * width - 1] = 0
    vertical = np.ones(size - width)
    degree = np.zeros(size)
    degree[:-1] += horizontal
    degree[1:] += horizontal
    degree[:-width] += vertical
    degree[width:] += vertical
    return sparse.diags(
        (-vertical, -horizontal, degree, -horizontal, -vertical),
        (-width, -1, 0, 1, width),
        shape=(size, size),
        format="csr",
    )


def _smooth_ridge(
    mosaic: np.ndarray,
    masks: np.ndarray,
    *,
    ridge: float = 0.05,
    rtol: float = 1e-7,
    atol: float = 1e-10,
    maxiter: int = 1000,
) -> np.ndarray:
    ridge = _real_parameter("ridge", ridge, zero_allowed=True)
    rtol = _real_parameter("rtol", rtol, zero_allowed=False)
    atol = _real_parameter("atol", atol, zero_allowed=True)
    maxiter = _positive_integer("maxiter", maxiter)
    output = _bilinear(mosaic, masks)
    laplacian = _grid_laplacian(mosaic.shape)
    flat_mosaic = mosaic.ravel()
    for channel in range(3):
        measured = masks[..., channel].ravel()
        missing = ~measured
        prior = output[..., channel].ravel().copy()
        matrix = laplacian[missing][:, missing]
        matrix = matrix + ridge * sparse.eye(matrix.shape[0], format="csr")
        rhs = ridge * prior[missing] - laplacian[missing][:, measured] @ flat_mosaic[measured]
        # SciPy 1.14 removed the old ``tol`` name in favor of ``rtol``.
        tolerance_name = "rtol" if "rtol" in inspect.signature(cg).parameters else "tol"
        solution, info = cg(
            matrix,
            rhs,
            x0=prior[missing],
            atol=atol,
            maxiter=maxiter,
            **{tolerance_name: rtol},
        )
        if info != 0 or not np.all(np.isfinite(solution)):
            raise ConvergenceError(
                f"smooth_ridge CG failed in {'RGB'[channel]} channel: info={info}, "
                f"maxiter={maxiter}; increase maxiter or relax rtol"
            )
        prior[missing] = solution
        prior[measured] = flat_mosaic[measured]
        output[..., channel] = prior.reshape(mosaic.shape)
    return output


def _gradient(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    dy, dx = np.zeros_like(values), np.zeros_like(values)
    dy[:-1] = values[1:] - values[:-1]
    dx[:, :-1] = values[:, 1:] - values[:, :-1]
    return dy, dx


def _gradient_adjoint(dy: np.ndarray, dx: np.ndarray) -> np.ndarray:
    result = np.zeros_like(dy)
    result[:-1] -= dy[:-1]
    result[1:] += dy[:-1]
    result[:, :-1] -= dx[:, :-1]
    result[:, 1:] += dx[:, :-1]
    return result


def _tv_objectives(
    values: np.ndarray,
    prior: np.ndarray,
    dual_y: np.ndarray,
    dual_x: np.ndarray,
    masks: np.ndarray,
    samples: np.ndarray,
    weight: float,
) -> tuple[float, float]:
    dy, dx = _gradient(values)
    primal = 0.5 * np.sum((values - prior) ** 2) + weight * np.sum(np.hypot(dy, dx))
    adjoint = _gradient_adjoint(dual_y, dual_x)
    # Exactly minimize G(v) + <K^T p, v> over the box and measured constraints.
    minimizer = np.clip(prior - adjoint, 0.0, 1.0)
    minimizer[masks] = samples
    dual = np.sum(0.5 * (minimizer - prior) ** 2 + adjoint * minimizer)
    return float(primal), float(dual)


def _tv(
    mosaic: np.ndarray,
    masks: np.ndarray,
    *,
    weight: float = 0.05,
    rtol: float = 1e-5,
    atol: float = 1e-8,
    maxiter: int = 2000,
    check_every: int = 10,
) -> np.ndarray:
    weight = _real_parameter("weight", weight, zero_allowed=True)
    rtol = _real_parameter("rtol", rtol, zero_allowed=False)
    atol = _real_parameter("atol", atol, zero_allowed=True)
    maxiter = _positive_integer("maxiter", maxiter)
    check_every = _positive_integer("check_every", check_every)
    prior = _bilinear(mosaic, masks)
    if weight == 0:
        return prior
    values, extrapolated = prior.copy(), prior.copy()
    samples = np.broadcast_to(mosaic[..., None], masks.shape)[masks]
    dual_y, dual_x = np.zeros_like(prior), np.zeros_like(prior)
    # ||K||^2 <= 8, so tau*sigma*||K||^2 < 1.
    tau = sigma = 0.35
    gap = float("inf")
    for iteration in range(1, maxiter + 1):
        dy, dx = _gradient(extrapolated)
        dual_y += sigma * dy
        dual_x += sigma * dx
        scale = np.maximum(1.0, np.hypot(dual_y, dual_x) / weight)
        dual_y /= scale
        dual_x /= scale
        previous = values
        values = np.clip(
            (values - tau * _gradient_adjoint(dual_y, dual_x) + tau * prior)
            / (1.0 + tau),
            0.0,
            1.0,
        )
        values[masks] = samples
        extrapolated = 2 * values - previous
        if iteration % check_every == 0 or iteration == maxiter:
            primal, dual = _tv_objectives(
                values, prior, dual_y, dual_x, masks, samples, weight
            )
            gap = primal - dual
            threshold = atol + rtol * max(abs(primal), abs(dual))
            if not np.isfinite(gap) or gap < -1e-10 * max(1.0, abs(primal)):
                raise ConvergenceError("tv encountered an invalid primal-dual gap")
            if gap <= threshold:
                return values
    raise ConvergenceError(
        f"tv did not converge in {maxiter} iterations: primal-dual gap={gap:.6g}; "
        "increase maxiter or relax rtol"
    )


def reconstruct(mosaic: np.ndarray, method: str, **kwargs) -> np.ndarray:
    """Reconstruct RGB from exported RGGB samples without modifying the input.

    ``method`` must be in :data:`METHODS`. Keyword parameters are method-specific
    and are documented in this module's docstring; unknown parameters raise
    TypeError. Invalid input values/shapes raise ValueError, and invalid dtypes
    raise TypeError. Iterative solver failures raise ConvergenceError.
    """
    mosaic = _validate_mosaic(mosaic)
    if not isinstance(method, str) or method not in METHODS:
        raise ValueError(f"unknown method {method!r}; expected one of {METHODS}")
    masks = _masks(mosaic.shape)
    implementations = {
        "nearest": _nearest,
        "bilinear": _bilinear,
        "smooth_ridge": _smooth_ridge,
        "tv": _tv,
    }
    output = implementations[method](mosaic, masks, **kwargs)
    np.clip(output, 0.0, 1.0, out=output)
    output[masks] = np.broadcast_to(mosaic[..., None], masks.shape)[masks]
    return output
