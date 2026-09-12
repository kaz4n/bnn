"""Tests for the Phase 0 probes.

These verify the probes and their CONTROLS behave as claimed. The controls are the
load-bearing part: a recovery number is only meaningful if the prior-only and shuffled
baselines are known to work, so most of these tests target the baselines rather than
the headline score.
"""
import numpy as np
import pytest

from experiments.rgb_sca import rgb_forward as F
from experiments.rgb_sca import rgb_probe as P


def _images(n=8, C=3, H=12, W=12, seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(n, C, H, W), dtype=np.uint8)


def _smooth_images(n=8, size=12, seed=0):
    """Low-frequency colour images: closer to photographs than uniform noise."""
    rng = np.random.default_rng(seed)
    out = []
    y, x = np.mgrid[0:size, 0:size] / size
    for _ in range(n):
        chans = []
        for _c in range(3):
            a, b, c0 = rng.uniform(-1, 1, 3)
            f = a * y + b * x + c0 * (y * x)
            f = (f - f.min()) / (np.ptp(f) or 1)
            chans.append((f * 255).astype(np.uint8))
        out.append(np.stack(chans))
    return np.stack(out)


# ------------------------------------------------------------------ distinguisher


def test_identity_permutation_never_differs():
    """Permuting by the identity must be a no-op; guards against a trivially-true probe."""
    img = _images(1)[0]
    ks = F.random_kernels(4, C=3, K=3, seed=1)
    for df in F.DATAFLOWS:
        r = P.distinguisher(img, ks, df, perm=(0, 1, 2))
        assert r["identical"] is True
        assert r["frac_differing"] == 0.0


def test_symmetric_kernels_hide_channels_in_accumulator_dataflows():
    """k[R]==k[G]==k[B] makes channel identity invisible where only sums are registered.

    This is the mechanism behind the distinguisher, so it is asserted directly: if this
    ever stops holding, the probe is measuring something other than what it claims.
    """
    img = _images(1)[0]
    sym = np.repeat(F.random_kernels(4, C=1, K=3, seed=2), 3, axis=1)
    assert P.kernel_channel_asymmetry(sym) == 0.0
    for df in ("parallel", "summed"):
        assert P.distinguisher(img, sym, df)["identical"] is True


def test_serial_separates_channels_even_with_symmetric_kernels():
    """Serial gets channel separability from cycle ordering, not from kernel weights."""
    img = _images(1)[0]
    sym = np.repeat(F.random_kernels(4, C=1, K=3, seed=3), 3, axis=1)
    assert P.distinguisher(img, sym, "serial")["identical"] is False


def test_asymmetric_kernels_expose_channels_in_all_dataflows():
    img = _images(1)[0]
    ks = F.random_kernels(9, C=3, K=3, seed=4)
    assert P.kernel_channel_asymmetry(ks) > 0.9
    for df in F.DATAFLOWS:
        assert P.distinguisher(img, ks, df)["identical"] is False


def test_greyscale_image_is_channel_permutation_invariant():
    """R==G==B means a channel permutation is a no-op on the data itself."""
    rng = np.random.default_rng(5)
    plane = rng.integers(0, 256, size=(12, 12), dtype=np.uint8)
    grey = np.stack([plane, plane, plane])
    ks = F.random_kernels(4, C=3, K=3, seed=6)
    for df in F.DATAFLOWS:
        assert P.distinguisher(grey, ks, df)["identical"] is True


# ------------------------------------------------------------------ features


def test_pixel_features_shape():
    power = np.arange(2 * 16 * 3, dtype=float).reshape(2, 48)
    feats = P.pixel_features(power, n_pixels=16, cycles_per_pixel=3, context=1)
    assert feats.shape == (16, 2 * 3 * 3)


def test_pixel_features_context_zero_uses_only_own_cycles():
    power = np.arange(1 * 4 * 2, dtype=float).reshape(1, 8)
    feats = P.pixel_features(power, n_pixels=4, cycles_per_pixel=2, context=0)
    assert feats.shape == (4, 2)
    assert np.array_equal(feats[0], np.array([0.0, 1.0]))


def test_build_dataset_shapes_and_targets():
    imgs = _images(3, H=8, W=8)
    ks = F.random_kernels(2, C=3, K=3, seed=7)
    X, Y, ids = P.build_dataset(imgs, ks, "serial")
    assert X.shape[0] == Y.shape[0] == ids.shape[0] == 3 * 64
    assert Y.shape[1] == 3
    # targets must be the actual centre pixels of image 0, row-major
    assert np.array_equal(Y[:64], imgs[0].reshape(3, 64).T)


# ------------------------------------------------------------------ recovery controls


@pytest.mark.parametrize("dataflow", F.DATAFLOWS)
def test_shuffled_control_collapses_to_prior(dataflow):
    """Real traces paired with wrong images must carry no advantage.

    This is the control that detects leakage through anything other than the trace.
    """
    imgs = _smooth_images(10, size=12, seed=8)
    ks = F.random_kernels(9, C=3, K=3, seed=9)
    r = P.recovery_probe(imgs, ks, dataflow, seed=0)
    prior = r["prior_only"]["mae_pooled"]
    shuf = r["shuffled"]["mae_pooled"]
    assert shuf == pytest.approx(prior, rel=0.10), (dataflow, prior, shuf)


def test_trace_beats_prior_on_serial_noise_free():
    """Serial at zero noise is the most favourable case; if it cannot beat the prior the
    probe is broken rather than the hypothesis being disproved."""
    imgs = _smooth_images(12, size=12, seed=10)
    ks = F.random_kernels(9, C=3, K=3, seed=11)
    r = P.recovery_probe(imgs, ks, "serial", seed=0)
    assert r["advantage_over_prior_pooled"] > 0


def test_advantage_is_monotone_nonincreasing_in_noise():
    """More noise must not help. Guards against a leak that noise would mask."""
    imgs = _smooth_images(12, size=12, seed=12)
    ks = F.random_kernels(9, C=3, K=3, seed=13)
    advs = [P.recovery_probe(imgs, ks, "serial", noise=nz, seed=0)
            ["advantage_over_prior_pooled"] for nz in (0.0, 1.0, 4.0)]
    assert advs[0] >= advs[1] >= advs[2] - 1e-9


def test_split_is_by_image_not_pixel():
    """Train and test must not share an image, or scores are inflated by correlation."""
    imgs = _smooth_images(6, size=8, seed=14)
    ks = F.random_kernels(2, C=3, K=3, seed=15)
    r = P.recovery_probe(imgs, ks, "summed", train_frac=0.5, seed=0)
    assert r["n_train_images"] == 3
    assert r["n_images"] == 6


def test_too_few_images_raises():
    imgs = _smooth_images(1, size=8, seed=16)
    ks = F.random_kernels(2, C=3, K=3, seed=17)
    with pytest.raises(ValueError):
        P.recovery_probe(imgs, ks, "summed", train_frac=1.0, seed=0)


def test_per_channel_scores_are_not_pooled_away():
    imgs = _smooth_images(6, size=8, seed=18)
    ks = F.random_kernels(2, C=3, K=3, seed=19)
    r = P.recovery_probe(imgs, ks, "serial", seed=0)
    assert len(r["trace"]["mae_per_channel"]) == 3
    assert len(r["advantage_over_prior_per_channel"]) == 3
