"""Tests for the 3-channel generator and its controls.

The controls are the load-bearing part -- a headline reconstruction number means nothing
without them -- so most of these target the controls and the colour metrics rather than
reconstruction quality.

The end-to-end test drives the generator from SIMULATED traces (the Phase 0 forward
model), so the pipeline and its controls are validated before any bitstream exists.
"""
import numpy as np
import pytest

from experiments.rgb_sca import rgb_forward as F
from experiments.rgb_sca import rgb_generator as G

torch = pytest.importorskip("torch")
nn = torch.nn


# ---------------------------------------------------------------- metrics


def test_perfect_reconstruction_scores_zero_error():
    rng = np.random.default_rng(0)
    truth = rng.random((4, 3, 8, 8))
    m = G.rgb_metrics(truth, truth)
    assert m["mae_pooled"] == pytest.approx(0.0, abs=1e-9)
    assert all(v == pytest.approx(0.0, abs=1e-9) for v in m["mae_per_channel"])
    assert m["luma_mae"] == pytest.approx(0.0, abs=1e-9)


def test_metrics_keep_channels_separate():
    """An error injected into one channel must not be smeared across the others."""
    truth = np.full((4, 3, 8, 8), 0.5)
    rec = truth.copy()
    rec[:, 0] += 0.2                                   # red only
    m = G.rgb_metrics(rec, truth)
    assert m["mae_per_channel"][0] > 40                # ~0.2*255
    assert m["mae_per_channel"][1] == pytest.approx(0.0, abs=1e-6)
    assert m["mae_per_channel"][2] == pytest.approx(0.0, abs=1e-6)


def test_luma_chroma_split_detects_colourisation():
    """Right structure, wrong colour: luma error small, chroma error large.

    This is the failure mode pooled PSNR hides and the reason the split exists.
    """
    rng = np.random.default_rng(1)
    y = rng.random((6, 8, 8))
    truth = np.stack([y * 1.0, y * 0.6, y * 0.2], axis=1)     # warm
    rec = np.stack([y * 0.2, y * 0.6, y * 1.0], axis=1)       # cool, same luminance-ish
    m = G.rgb_metrics(rec, truth)
    assert m["chroma_over_luma"] > 1.0, m


def test_grey_input_has_near_zero_chroma():
    rng = np.random.default_rng(2)
    y = rng.random((4, 8, 8))
    grey = np.stack([y, y, y], axis=1)
    m = G.rgb_metrics(grey, grey)
    assert m["chroma_mae"][0] == pytest.approx(0.0, abs=1e-6)


def test_metrics_clip_out_of_range_predictions():
    truth = np.zeros((2, 3, 8, 8))
    rec = np.full((2, 3, 8, 8), 5.0)          # far outside [0,1]
    m = G.rgb_metrics(rec, truth)
    assert m["mae_pooled"] == pytest.approx(255.0, rel=1e-6)


# ---------------------------------------------------------------- permutation control


def test_channel_swap_penalty_zero_for_grey_reconstruction():
    """A grey output fits either channel assignment equally, so the penalty is zero.

    This is exactly the case the control must flag, and the case a pooled score would
    happily call a success.
    """
    rng = np.random.default_rng(3)
    y = rng.random((5, 8, 8))
    truth = np.stack([y, y, y], axis=1)
    rec = np.stack([y, y, y], axis=1)
    cp = G.channel_permutation_check(rec, truth)
    assert cp["swap_penalty"] == pytest.approx(0.0, abs=1e-9)


def test_channel_swap_penalty_positive_when_colour_is_real():
    rng = np.random.default_rng(4)
    truth = rng.random((5, 3, 8, 8))
    cp = G.channel_permutation_check(truth, truth)     # perfect recovery
    assert cp["swap_penalty"] > 0
    assert cp["swap_penalty_ratio"] > 1.0


# ---------------------------------------------------------------- arms


def test_prior_only_is_analytic_not_trained():
    """prior_only must be the closed-form optimal constant, never a trained network.

    Regression test. The trained version scored MAE 122 where the optimal constant scores
    54, because identical inputs give zero feature variance and BatchNorm cannot train on
    that. It silently inflated the trace arm's apparent advantage by ~68 MAE.
    """
    with pytest.raises(ValueError, match="analytic"):
        G.make_arm_features(np.zeros((4, 3)), "prior_only", np.random.default_rng(0))


def test_analytic_prior_equals_training_mean_and_beats_any_other_constant():
    rng = np.random.default_rng(5)
    y = rng.random((40, 3, 8, 8))
    tr, te = y[:30], y[30:]
    pred = G.analytic_prior(tr, len(te))
    assert pred.shape == te.shape
    assert np.allclose(pred[0], tr.mean(0))
    # No other constant can beat the mean under squared error.
    best = float(np.mean((pred - te) ** 2))
    for shift in (-0.1, 0.05, 0.2):
        other = np.clip(pred + shift, 0, 1)
        assert float(np.mean((other - te) ** 2)) >= best - 1e-12


def test_prior_only_arm_scores_near_the_analytic_optimum():
    """End-to-end guard: the prior arm's reported MAE must match the closed-form value.

    This is the check that would have caught the 122-vs-54 failure.
    """
    rng = np.random.default_rng(11)
    imgs = (rng.random((40, 3, 8, 8)) * 255).astype(np.uint8)
    feats = rng.random((40, 6)).astype(np.float32)
    res, rec, truth, te = G.train_arm(feats, imgs, "prior_only", _Args(), torch, nn,
                                      np.random.default_rng(0))
    y = imgs.astype(np.float64) / 255.0
    mask = np.ones(len(y), bool); mask[te] = False
    expected = float(np.abs(y[te] - y[mask].mean(0)) .mean() * 255)
    assert res["mae_pooled"] == pytest.approx(expected, rel=1e-6)


def test_shuffled_arm_preserves_marginals_but_breaks_pairing():
    rng = np.random.default_rng(6)
    feats = rng.random((20, 5))
    out = G.make_arm_features(feats, "shuffled", rng)
    assert np.allclose(np.sort(out, axis=0), np.sort(feats, axis=0))
    assert not np.allclose(out, feats)


def test_trace_arm_is_identity():
    rng = np.random.default_rng(7)
    feats = rng.random((6, 4))
    assert np.allclose(G.make_arm_features(feats, "trace", rng), feats)


def test_unknown_arm_raises():
    with pytest.raises(ValueError):
        G.make_arm_features(np.zeros((2, 2)), "wishful", np.random.default_rng(0))


# ---------------------------------------------------------------- featurization


def _manifest(dwell=8, spc=4):
    return {"samples_per_cycle": spc, "dwell": dwell}


def test_settle_window_is_a_strict_subset_of_the_dwell():
    rng = np.random.default_rng(8)
    tr = rng.random((3, 8 * 4 * 10))
    man = _manifest()
    f_all = G.featurize_rgb(tr, man, "window_abs")
    f_set = G.featurize_rgb(tr, man, "settle")
    assert f_all.shape == f_set.shape
    assert not np.allclose(f_all, f_set)


def test_featurize_scales_with_dwell():
    rng = np.random.default_rng(9)
    tr = rng.random((2, 16 * 4 * 5))
    f = G.featurize_rgb(tr, _manifest(dwell=16), "settle")
    assert f.shape == (2, 5)


def test_unknown_feature_mode_raises():
    with pytest.raises(ValueError):
        G.featurize_rgb(np.zeros((2, 128)), _manifest(), "vibes")


# ---------------------------------------------------------------- model


def test_model_emits_three_channels_at_the_requested_size():
    m = G.build_rgb_model(in_dim=16, out_side=32, torch=torch, nn=nn)
    m.eval()
    with torch.no_grad():
        y = m(torch.zeros(2, 16))
    assert y.shape == (2, 3, 32, 32)
    assert float(y.min()) >= 0.0 and float(y.max()) <= 1.0


def test_model_rejects_unsupported_output_size():
    with pytest.raises(ValueError):
        G.build_rgb_model(in_dim=8, out_side=30, torch=torch, nn=nn)


def test_gradmse_is_zero_for_identical_inputs():
    a = torch.rand(2, 3, 8, 8)
    assert float(G.gradmse(a, a, torch, nn.MSELoss())) == pytest.approx(0.0, abs=1e-12)


# ---------------------------------------------------------------- end to end


class _Args:
    epochs = 6
    batch = 16
    lr = 3e-3
    loss = "gradmse"
    test_frac = 0.25
    seed = 0


def test_end_to_end_on_simulated_traces_controls_behave():
    """Drive the real training path from Phase 0 simulated traces.

    Asserts the three properties that make a hardware result interpretable:
      1. the trace arm beats prior-only,
      2. the shuffled arm does NOT (it has no usable information),
      3. prior-only and shuffled land close to each other.
    Reconstruction quality itself is not asserted -- this is a pipeline test.
    """
    rng = np.random.default_rng(0)
    n, side = 48, 8
    # Smooth colour images: a constant-ish image would make the prior arm unbeatable.
    yy, xx = np.mgrid[0:side, 0:side] / side
    imgs = []
    for _ in range(n):
        chans = []
        for _c in range(3):
            a, b, c0 = rng.uniform(-1, 1, 3)
            f = a * yy + b * xx + c0 * yy * xx
            f = (f - f.min()) / (np.ptp(f) or 1)
            chans.append((f * 255).astype(np.uint8))
        imgs.append(np.stack(chans))
    images = np.stack(imgs)

    kernels = F.random_kernels(4, C=3, K=3, seed=1)
    feats = np.stack([F.power_features(im, kernels, "summed").ravel() for im in images])
    feats = feats.astype(np.float32)

    res, _ = G.run_all_arms(feats, images, _Args(), torch, nn)
    trace_mae = res["trace"]["mae_pooled"]
    prior_mae = res["prior_only"]["mae_pooled"]
    shuf_mae = res["shuffled"]["mae_pooled"]

    assert trace_mae < prior_mae, (trace_mae, prior_mae)
    assert shuf_mae > trace_mae, (shuf_mae, trace_mae)
    assert abs(shuf_mae - prior_mae) < 0.5 * prior_mae
    assert len(res["trace"]["mae_per_channel"]) == 3
    assert "channel_permutation" in res["trace"]
    assert "chroma_advantage" in res["summary"]
