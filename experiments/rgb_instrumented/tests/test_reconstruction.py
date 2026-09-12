"""Focused correctness checks for reconstruction of exported Bayer samples."""

import numpy as np
import pytest
from scipy import optimize

from experiments.rgb_instrumented import reconstruction as rec


def mosaic_from_rgb(rgb):
    mosaic = rgb[..., 1].copy()
    mosaic[::2, ::2] = rgb[::2, ::2, 0]
    mosaic[1::2, 1::2] = rgb[1::2, 1::2, 2]
    return mosaic


@pytest.mark.parametrize("method", rec.METHODS)
@pytest.mark.parametrize("shape", [(2, 2), (3, 5), (8, 10)])
def test_constant_color_and_channel_order(method, shape):
    rgb = np.broadcast_to([0.12, 0.53, 0.91], (*shape, 3)).copy()
    result = rec.reconstruct(mosaic_from_rgb(rgb), method)
    np.testing.assert_allclose(result, rgb, rtol=0, atol=1e-12)


@pytest.mark.parametrize("method", rec.METHODS)
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_exact_measurements_range_shape_and_no_input_mutation(method, dtype):
    mosaic = np.random.default_rng(19).random((9, 11)).astype(dtype)
    original = mosaic.copy()
    result = rec.reconstruct(mosaic, method)
    assert result.shape == (9, 11, 3)
    assert result.dtype == np.float64
    assert np.all(np.isfinite(result))
    assert np.all((result >= 0) & (result <= 1))
    np.testing.assert_array_equal(result[::2, ::2, 0], mosaic[::2, ::2])
    np.testing.assert_array_equal(result[1::2, 1::2, 2], mosaic[1::2, 1::2])
    np.testing.assert_array_equal(result[::2, 1::2, 1], mosaic[::2, 1::2])
    np.testing.assert_array_equal(result[1::2, ::2, 1], mosaic[1::2, ::2])
    np.testing.assert_array_equal(mosaic, original)


def test_bilinear_expected_local_averages():
    mosaic = np.zeros((5, 5))
    mosaic[0, 0], mosaic[0, 2] = 0.2, 0.4
    mosaic[2, 0], mosaic[2, 2] = 0.6, 0.8
    mosaic[0, 1], mosaic[1, 0] = 0.1, 0.3
    result = rec.reconstruct(mosaic, "bilinear")
    assert result[0, 1, 0] == pytest.approx(0.3)
    assert result[1, 1, 0] == pytest.approx(0.5)
    assert result[0, 0, 1] == pytest.approx(0.2)


def test_nearest_only_uses_closest_same_color_samples():
    mosaic = np.random.default_rng(2).random((5, 6))
    result = rec.reconstruct(mosaic, "nearest")
    for channel, parity in enumerate(((0, 0), None, (1, 1))):
        coordinates = [
            (y, x)
            for y in range(5)
            for x in range(6)
            if ((y % 2 != x % 2) if parity is None else (y % 2, x % 2) == parity)
        ]
        for y in range(5):
            for x in range(6):
                distances = np.array([(cy-y)**2 + (cx-x)**2 for cy, cx in coordinates])
                nearest_values = [mosaic[coordinates[i]] for i in np.flatnonzero(distances == distances.min())]
                assert result[y, x, channel] in nearest_values


@pytest.mark.parametrize("shape", [(0, 3), (1, 4), (4, 1), (5,), (2, 2, 3), ()])
def test_invalid_dimensions(shape):
    with pytest.raises(ValueError, match="shape"):
        rec.reconstruct(np.zeros(shape, dtype=float), "bilinear")


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf, -0.01, 1.01])
def test_invalid_values(value):
    mosaic = np.full((2, 2), 0.5)
    mosaic[0, 0] = value
    with pytest.raises(ValueError):
        rec.reconstruct(mosaic, "bilinear")


@pytest.mark.parametrize("value", [[[0.0, 0.0], [0.0, 0.0]], np.zeros((2, 2), dtype=int), np.zeros((2, 2), dtype=complex), np.zeros((2, 2), dtype=bool)])
def test_invalid_input_types(value):
    with pytest.raises(TypeError):
        rec.reconstruct(value, "bilinear")


def test_unknown_methods_and_parameters():
    mosaic = np.zeros((2, 2))
    assert rec.get_methods() == ("nearest", "bilinear", "smooth_ridge", "tv")
    with pytest.raises(ValueError, match="unknown method"):
        rec.reconstruct(mosaic, "cubic")
    with pytest.raises(TypeError):
        rec.reconstruct(mosaic, "bilinear", weight=0.1)


@pytest.mark.parametrize("method,kwargs", [("smooth_ridge", {"ridge": -1}), ("smooth_ridge", {"rtol": 0}), ("tv", {"weight": np.inf}), ("tv", {"maxiter": 0}), ("tv", {"check_every": -1}), ("tv", {"atol": -1})])
def test_invalid_solver_parameters(method, kwargs):
    with pytest.raises(ValueError):
        rec.reconstruct(np.zeros((2, 2)), method, **kwargs)


@pytest.mark.parametrize("method", ["smooth_ridge", "tv"])
def test_noninteger_iteration_limit(method):
    with pytest.raises(TypeError):
        rec.reconstruct(np.zeros((2, 2)), method, maxiter=2.5)


def test_smooth_ridge_satisfies_discrete_normal_equations():
    mosaic = np.random.default_rng(5).random((5, 6))
    ridge = 0.13
    prior = rec.reconstruct(mosaic, "bilinear")
    result = rec.reconstruct(mosaic, "smooth_ridge", ridge=ridge, rtol=1e-10)
    for channel in range(3):
        for y in range(5):
            for x in range(6):
                measured_channel = 0 if y % 2 == x % 2 == 0 else 2 if y % 2 == x % 2 == 1 else 1
                if channel == measured_channel:
                    continue
                neighbors = [(yy, xx) for yy, xx in ((y-1, x), (y+1, x), (y, x-1), (y, x+1)) if 0 <= yy < 5 and 0 <= xx < 6]
                residual = sum(result[y, x, channel] - result[yy, xx, channel] for yy, xx in neighbors)
                residual += ridge * (result[y, x, channel] - prior[y, x, channel])
                assert abs(residual) < 2e-9


@pytest.mark.parametrize("info", [3, -1])
def test_cg_failure_is_reported(monkeypatch, info):
    def failed_cg(matrix, rhs, **kwargs):
        return np.zeros_like(rhs), info
    monkeypatch.setattr(rec, "cg", failed_cg)
    with pytest.raises(rec.ConvergenceError, match="CG failed"):
        rec.reconstruct(np.full((4, 4), 0.5), "smooth_ridge")


def test_cg_old_tolerance_api(monkeypatch):
    real_cg = rec.cg
    calls = []
    def old_cg(matrix, rhs, *, x0, atol, maxiter, tol):
        calls.append(tol)
        return real_cg(matrix, rhs, x0=x0, atol=atol, maxiter=maxiter, rtol=tol)
    monkeypatch.setattr(rec, "cg", old_cg)
    rec.reconstruct(np.random.default_rng(8).random((4, 4)), "smooth_ridge")
    assert calls == [1e-7] * 3


def test_tv_exhausted_budget_is_reported():
    mosaic = np.random.default_rng(31).random((8, 9))
    with pytest.raises(rec.ConvergenceError, match="did not converge"):
        rec.reconstruct(mosaic, "tv", maxiter=1, rtol=1e-14, atol=0)


def test_tv_zero_weight_is_bilinear():
    mosaic = np.random.default_rng(13).random((5, 8))
    np.testing.assert_array_equal(rec.reconstruct(mosaic, "tv", weight=0), rec.reconstruct(mosaic, "bilinear"))


def test_tv_matches_independent_small_convex_optimization():
    mosaic = np.array([[0.1, 0.8, 0.4], [0.6, 0.9, 0.2], [0.7, 0.3, 0.5]])
    weight = 0.08
    prior = rec.reconstruct(mosaic, "bilinear")
    result = rec.reconstruct(mosaic, "tv", weight=weight, rtol=1e-9, atol=1e-11, maxiter=10000)
    mask = np.zeros((3, 3), dtype=bool)
    mask[::2, ::2] = True
    def objective(unknown):
        channel = prior[..., 0].copy()
        channel[~mask] = unknown
        dy = np.vstack((np.diff(channel, axis=0), np.zeros((1, 3))))
        dx = np.hstack((np.diff(channel, axis=1), np.zeros((3, 1))))
        return 0.5 * np.sum((channel - prior[..., 0])**2) + weight * np.sum(np.hypot(dy, dx))
    reference = optimize.minimize(objective, prior[..., 0][~mask], method="Powell", bounds=[(0, 1)] * np.count_nonzero(~mask), options={"ftol": 1e-12, "xtol": 1e-12, "maxiter": 10000})
    assert reference.success
    assert objective(result[..., 0][~mask]) <= reference.fun + 1e-7
    assert objective(result[..., 0][~mask]) < objective(prior[..., 0][~mask])
