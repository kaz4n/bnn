"""Tests for the calibrated-validation machinery.

These check the properties the methodology actually relies on. Two of them matter more
than the rest, and both encode a mistake this project has already made once:

  * a contract violation must RAISE, not be quietly repaired -- a coerced array is how an
    invalid result acquires a confident-looking number;
  * a split must be checked for scene overlap, not assumed disjoint -- the Phase 0 ranking
    was retracted because overlapping scenes inverted the result.

The conformal tests are property tests against the finite-sample guarantee itself, not
regression tests against numbers this code happened to produce.
"""
import math

import numpy as np
import pytest

from experiments.rgb_validity import calibration as C
from experiments.rgb_validity import evaluate_frozen as E
from experiments.rgb_validity import source_audit as SA
from experiments.rgb_validity import tasks as T
from experiments.rgb_validity.schema import ContractError, FrozenPredictionBatch


def _batch(n=10, h=8, w=8, seed=0, **over):
    rng = np.random.default_rng(seed)
    kw = dict(mode="summed", group="natural",
              truth=rng.random((n, 3, h, w)),
              prediction=rng.random((n, 3, h, w)),
              prior_prediction=rng.random((n, 3, h, w)),
              source_id=np.arange(n), scene_id=np.arange(n), session_id=None)
    kw.update(over)
    return FrozenPredictionBatch(**kw)


# ---------------------------------------------------------------- schema contracts

def test_valid_batch_constructs():
    assert _batch().validate() is not None


@pytest.mark.parametrize("over,frag", [
    ({"prediction": np.zeros((10, 3, 8, 9))}, "shape mismatch"),
    ({"truth": np.zeros((10, 4, 8, 8))}, "expected (N,3,H,W)"),
    ({"source_id": np.arange(3)}, "source_id"),
    ({"scene_id": np.arange(3)}, "scene_id"),
    ({"session_id": np.arange(3)}, "session_id"),
])
def test_contract_violations_raise(over, frag):
    with pytest.raises(ContractError) as ei:
        _batch(**over)
    assert frag in str(ei.value)


def test_non_finite_truth_raises():
    t = np.zeros((4, 3, 8, 8))
    t[0, 0, 0, 0] = np.nan
    with pytest.raises(ContractError, match="non-finite"):
        _batch(n=4, truth=t)


def test_out_of_range_truth_raises_but_prediction_does_not():
    """Truth must be a valid image; predictions may exceed [0,1] and stay visible."""
    with pytest.raises(ContractError, match=r"outside \[0,1\]"):
        _batch(n=4, truth=np.full((4, 3, 8, 8), 1.5))
    b = _batch(n=4, prediction=np.full((4, 3, 8, 8), 1.5))
    assert b.n_clipped == 4 * 3 * 8 * 8


def test_hashes_change_with_content():
    a, b = _batch(seed=0), _batch(seed=1)
    assert a.hashes()["prediction"] != b.hashes()["prediction"]
    assert a.hashes()["truth"] == _batch(seed=0).hashes()["truth"]


# ---------------------------------------------------------------- task transforms

def test_grey_image_has_zero_chroma():
    g = np.repeat(np.random.default_rng(0).random((5, 1, 8, 8)), 3, axis=1)
    assert np.allclose(T.task_Cb(g), 0, atol=1e-12)
    assert np.allclose(T.task_Cr(g), 0, atol=1e-12)


def test_identical_prediction_scores_zero_on_every_task():
    x = np.random.default_rng(0).random((5, 3, 8, 8))
    for name in T.ALL:
        assert np.allclose(T.task_scores(x, x, name), 0.0)


def test_task_scores_are_in_0_255_units():
    """A constant offset d in every pixel must score 255*d on Y, by construction."""
    t = np.full((3, 3, 8, 8), 0.5)
    p = np.full((3, 3, 8, 8), 0.6)
    assert np.allclose(T.task_scores(p, t, "Y"), 255 * 0.1, atol=1e-9)


def test_coarse_task_hides_error_that_full_resolution_sees():
    """Why the coarse task is SECONDARY: block-mean-zero error is invisible to it.

    A checkerboard of +d/-d inside every 4x4 block averages to exactly zero, so the 8x8
    task scores it perfect while the full-resolution task scores 255*d. A method can look
    good on block means and be wrong at every pixel.
    """
    d = 0.1
    t = np.full((4, 3, 8, 8), 0.5)
    sign = np.ones((8, 8))
    sign[0::2, 1::2] = sign[1::2, 0::2] = -1.0
    p = t + d * sign
    assert np.allclose(T.task_scores(p, t, "Y_coarse8"), 0.0, atol=1e-9)
    assert np.allclose(T.task_scores(p, t, "Y"), 255 * d, atol=1e-9)


def test_coarse_block_must_divide():
    with pytest.raises(ValueError, match="not divisible"):
        T.task_Y_coarse(np.zeros((2, 3, 7, 7)))


def test_primary_and_secondary_are_disjoint_and_declared():
    assert set(T.PRIMARY) == {"Y", "Cb", "Cr"}
    assert set(T.SECONDARY) == {"Y_coarse8"}
    assert not set(T.PRIMARY) & set(T.SECONDARY)


# ---------------------------------------------------------------- conformal guarantee

def test_radius_is_kth_order_statistic():
    s = np.arange(1.0, 101.0)
    q = C.conformal_radius(s, 0.10)
    assert q["k"] == math.ceil(101 * 0.9) == 91
    assert q["radius"] == 91.0


def test_insufficient_calibration_yields_infinite_radius_not_the_max():
    """8 examples cannot support 90%. Substituting max(scores) fabricates a bound."""
    s = np.arange(1.0, 9.0)
    q = C.conformal_radius(s, 0.10)
    assert q["k"] == 9 and not q["finite"]
    assert q["radius"] == float("inf") and q["radius"] != s.max()


def test_empirical_coverage_meets_the_guarantee_on_exchangeable_data():
    """The property the whole method rests on: >= 1-alpha marginal coverage."""
    rng = np.random.default_rng(0)
    alpha, trials = 0.10, 2000
    rates = []
    for _ in range(trials):
        s = rng.standard_exponential(200)  # skewed on purpose; no distributional claim
        radius = C.conformal_radius(s[:100], alpha)["radius"]
        rates.append(float(np.mean(s[100:] <= radius)))
    assert np.mean(rates) >= 1 - alpha - 0.02


def test_coverage_is_exact_enough_to_be_informative():
    """Not merely valid -- a radius of inf would also be 'valid' and say nothing."""
    rng = np.random.default_rng(1)
    cov = np.mean([
        (rng.standard_normal(100) <= C.conformal_radius(
            rng.standard_normal(400), 0.10)["radius"]).mean() for _ in range(400)])
    assert 0.86 < cov < 0.94


def test_wilson_interval_brackets_the_rate_and_flags_undercoverage():
    good = C.evaluate_coverage(1.0, np.zeros(100), 0.10)
    assert good["coverage"] == 1.0 and not good["materially_undercovers"]
    bad = C.evaluate_coverage(0.0, np.ones(100), 0.10)
    assert bad["coverage"] == 0.0 and bad["materially_undercovers"]
    lo, hi = bad["wilson_95"]
    assert 0.0 <= lo <= hi <= 1.0


def test_undercoverage_flag_needs_evidence_not_just_a_low_point_estimate():
    """With 10 examples the interval is wide, so 0.8 must NOT be called undercoverage."""
    s = np.array([0.0] * 8 + [2.0] * 2)
    r = C.evaluate_coverage(1.0, s, 0.10)
    assert r["coverage"] == 0.8
    assert not r["materially_undercovers"]


def test_empty_calibration_raises():
    with pytest.raises(ValueError, match="no calibration"):
        C.conformal_radius(np.array([]), 0.10)


# ---------------------------------------------------------------- splitting

def test_scene_disjoint_split_shares_no_scene():
    scenes = np.repeat(np.arange(20), 3)  # 3 views per scene
    cal, ass = E.split_by_scene(scenes, 0.5, seed=0)
    assert len(cal) + len(ass) == len(scenes)
    assert not set(scenes[cal]) & set(scenes[ass])


def test_split_is_deterministic_under_seed():
    s = np.arange(40)
    assert np.array_equal(E.split_by_scene(s, 0.5, 7)[0], E.split_by_scene(s, 0.5, 7)[0])
    assert not np.array_equal(E.split_by_scene(s, 0.5, 7)[0],
                              E.split_by_scene(s, 0.5, 8)[0])


def test_row_level_split_of_repeated_scenes_is_rejected():
    """The Phase 0 failure, encoded: a row split leaks scenes and must be caught."""
    scenes = np.repeat(np.arange(10), 2)
    with pytest.raises(ValueError, match="share"):
        SA.assert_scene_disjoint(scenes[0::2], scenes[1::2])


def test_assert_scene_disjoint_passes_on_clean_split():
    assert SA.assert_scene_disjoint(np.arange(5), np.arange(5, 10)) is True


# ---------------------------------------------------------------- decision rule

def _cell(**tasks):
    return {"tasks": tasks}


def _task(useful, primary=True):
    return {"primary": primary, "useful": useful}


def test_decision_requires_both_coverage_and_tightness():
    d = E.decide(_cell(Y=_task(True), Cb=_task(False), Cr=_task(None)))
    assert d["established_primary_tasks"] == ["Y"]
    assert sorted(d["withheld_primary_tasks"]) == ["Cb", "Cr"]


def test_secondary_tasks_never_enter_the_verdict():
    d = E.decide(_cell(Y=_task(False), Y_coarse8=_task(True, primary=False)))
    assert d["established_primary_tasks"] == []
    assert "Y_coarse8" not in d["withheld_primary_tasks"]


def test_usefulness_is_false_when_covered_but_wider_than_prior():
    """A set that covers only because it is huge is not evidence of recovery."""
    e = np.full(200, 10.0)
    ep = np.full(200, 5.0)
    cal, ass = np.arange(100), np.arange(100, 200)
    q = C.conformal_radius(e[cal], 0.10)
    qp = C.conformal_radius(ep[cal], 0.10)
    cov = C.evaluate_coverage(q["radius"], e[ass], 0.10)
    assert cov["coverage"] == 1.0                 # covers perfectly
    assert not (q["radius"] < qp["radius"])       # but is wider than the prior


# ---------------------------------------------------------------- seed stability

def _stab_batch(n=40, gain=1.0, seed=0):
    """Prediction is truth plus noise scaled by `gain`; prior is truth plus fixed noise.

    gain < 1 makes the prediction genuinely better than the prior, gain == 1 makes them
    statistically identical.
    """
    rng = np.random.default_rng(seed)
    t = rng.random((n, 3, 8, 8)) * 0.5 + 0.25
    return FrozenPredictionBatch(
        mode="summed", group="g", truth=t,
        prediction=np.clip(t + gain * 0.05 * rng.standard_normal(t.shape), 0, 1),
        prior_prediction=np.clip(t + 0.05 * rng.standard_normal(t.shape), 0, 1),
        source_id=np.arange(n), scene_id=np.arange(n), session_id=None)


def test_stability_reports_one_row_per_task_with_ranges():
    r = E.seed_stability(_stab_batch(), 0.10, 0.5, range(4))
    assert set(r) == set(T.ALL)
    for row in r.values():
        assert row["n_seeds"] == 4
        assert row["coverage_min_max"][0] <= row["coverage_min_max"][1]
        assert 0.0 <= row["fraction_of_seeds_useful"] <= 1.0


def test_a_clearly_better_predictor_is_useful_in_every_split():
    # n matches the real pilot. At n=40 the calibration set is 20 examples and the
    # coverage gate is dominated by split noise -- which is exactly what this diagnostic
    # exists to expose, so it must not be what this test measures.
    r = E.seed_stability(_stab_batch(n=200, gain=0.1), 0.10, 0.5, range(8))
    assert r["Y"]["fraction_of_seeds_useful"] == 1.0
    assert r["Y"]["stable"] is True
    assert r["Y"]["ratio_range_excludes_one"] is True


def test_an_equivalent_predictor_does_not_hold_across_splits():
    """gain==1: prediction and prior are the same quality, so the verdict must waver."""
    r = E.seed_stability(_stab_batch(n=200, gain=1.0), 0.10, 0.5, range(20))
    assert 0.0 <= r["Y"]["fraction_of_seeds_useful"] < 1.0
    assert r["Y"]["ratio_range_excludes_one"] is False


def test_stability_never_touches_the_predictions():
    b = _stab_batch()
    before = b.hashes()
    E.seed_stability(b, 0.10, 0.5, range(4))
    assert b.hashes() == before


def test_high_stability_fraction_is_not_claimed_as_strength_of_evidence():
    """The misreading this diagnostic invites, encoded.

    A predictor a hair better than the prior fires the rule in most splits while its
    ratio range still straddles 1.0. The row must expose both facts, so a high fraction
    cannot be quoted as support on its own.
    """
    r = E.seed_stability(_stab_batch(n=200, gain=0.97, seed=3), 0.10, 0.5,
                         range(20))["Y"]
    assert r["ratio_range_excludes_one"] is False
    assert "NOT strength of evidence" in r["note"]
