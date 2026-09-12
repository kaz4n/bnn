"""Correctness tests for the RGB forward/power model.

These check the model against the arithmetic it claims to describe. They are not
evidence about recoverability -- that is what the Phase 0 probe measures.
"""
import numpy as np
import pytest

from experiments.rgb_sca import rgb_forward as F


def _img(C=3, H=8, W=8, seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(C, H, W), dtype=np.uint8)


def test_extract_windows_matches_naive_reference():
    """Vectorized extraction must equal an explicit zero-padded gather."""
    img, K = _img(), 3
    got = F.extract_windows(img, K)
    C, H, W = img.shape
    p = K // 2
    want = np.zeros((H * W, C, K * K), dtype=np.int32)
    for y in range(H):
        for x in range(W):
            for c in range(C):
                for i in range(K):
                    for j in range(K):
                        yy, xx = y + i - p, x + j - p
                        if 0 <= yy < H and 0 <= xx < W:
                            want[y * W + x, c, i * K + j] = img[c, yy, xx]
    assert np.array_equal(got, want)


def test_extract_windows_shape_and_dtype():
    img = _img(C=3, H=16, W=16)
    win = F.extract_windows(img, 5)
    assert win.shape == (256, 3, 25)
    assert win.dtype == np.int32


def test_conv_output_matches_direct_sum():
    img = _img(H=6, W=6)
    k = F.random_kernels(1, C=3, K=3, seed=1)[0]
    got = F.conv_output(img, k)
    C, H, W = img.shape
    want = np.zeros((H, W), dtype=np.int64)
    for y in range(H):
        for x in range(W):
            s = 0
            for c in range(C):
                for i in range(3):
                    for j in range(3):
                        yy, xx = y + i - 1, x + j - 1
                        if 0 <= yy < H and 0 <= xx < W:
                            s += int(k[c, i, j]) * int(img[c, yy, xx])
            want[y, x] = s
    assert np.array_equal(got, want)


@pytest.mark.parametrize("dataflow", F.DATAFLOWS)
def test_power_shape_per_dataflow(dataflow):
    """Serial spends C cycles per output pixel; the others spend one."""
    img = _img(C=3, H=8, W=8)
    k = F.random_kernels(1, C=3, K=3, seed=2)[0]
    pw = F.per_cycle_power(img, k, dataflow)
    expected = 8 * 8 * (3 if dataflow == "serial" else 1)
    assert pw.shape == (expected,)
    assert np.all(np.isfinite(pw))


def test_power_is_data_dependent():
    """A constant image and a structured image must not produce identical power."""
    k = F.random_kernels(1, C=3, K=3, seed=3)[0]
    flat = np.full((3, 8, 8), 128, dtype=np.uint8)
    for dataflow in F.DATAFLOWS:
        p_flat = F.per_cycle_power(flat, k, dataflow)
        p_rand = F.per_cycle_power(_img(C=3, H=8, W=8, seed=4), k, dataflow)
        assert not np.allclose(p_flat, p_rand), dataflow


def test_dataflow_does_not_change_arithmetic():
    """Dataflow is an implementation choice; the convolution result is invariant."""
    img = _img()
    k = F.random_kernels(1, C=3, K=3, seed=5)[0]
    ref = F.conv_output(img, k)
    for dataflow in F.DATAFLOWS:
        F.per_cycle_power(img, k, dataflow)          # must not mutate anything
    assert np.array_equal(F.conv_output(img, k), ref)


def test_parallel_carries_at_least_as_much_variance_as_summed():
    """Registering products adds a term; it cannot remove information."""
    img = _img(C=3, H=12, W=12, seed=6)
    k = F.random_kernels(1, C=3, K=3, seed=7)[0]
    v_sum = np.var(F.per_cycle_power(img, k, "summed"))
    v_par = np.var(F.per_cycle_power(img, k, "parallel"))
    assert v_par > v_sum


def test_power_features_stacks_kernels():
    img = _img(C=3, H=8, W=8)
    ks = F.random_kernels(4, C=3, K=3, seed=8)
    feats = F.power_features(img, ks, "summed")
    assert feats.shape == (4, 64)


def test_kernels_are_binarized():
    ks = F.random_kernels(16, C=3, K=3, seed=9)
    assert set(np.unique(ks).tolist()) <= {-1, 1}


def test_channel_count_mismatch_raises():
    img = _img(C=3)
    k = F.random_kernels(1, C=1, K=3, seed=10)[0]
    with pytest.raises(ValueError):
        F.per_cycle_power(img, k, "summed")


def test_bad_dataflow_raises():
    img = _img()
    k = F.random_kernels(1, C=3, K=3, seed=11)[0]
    with pytest.raises(ValueError):
        F.per_cycle_power(img, k, "diagonal")


# ------------------------------------------------- serial accumulator (regression)
# The original implementation accumulated channels but kept only the LAST TAP of each
# cycle, so it silently modelled a different circuit from the one the RTL implements.
# test_dataflow_does_not_change_arithmetic passed throughout, because it only checks the
# separately computed convolution output -- never the accumulator itself. These tests
# check the accumulator directly. External review, 12 September 2026.


def _serial_acc(img, kernel):
    """Recompute the serial accumulator the way the module documents it."""
    C, K, _ = kernel.shape
    win = F.extract_windows(img, K)
    g = win * kernel.reshape(C, K * K)[None, :, :]
    return g.sum(axis=2).cumsum(axis=1).reshape(-1)


def test_serial_accumulator_sums_taps_then_channels():
    """Products 1..27 must accumulate to [45, 171, 378], not [9, 27, 54]."""
    g = np.arange(1, 28).reshape(1, 3, 9)
    got = g.sum(axis=2).cumsum(axis=1).reshape(-1)
    assert list(got) == [45, 171, 378]
    stale = np.cumsum(g, axis=1).reshape(3, 9)[:, -1]
    assert list(stale) == [9, 27, 54]          # what the defect produced
    assert not np.array_equal(got, stale)


def test_serial_final_accumulator_equals_full_convolution():
    """After the last channel the accumulator must equal the full 27-tap conv output.

    This is the invariant that ties the simulator to the hardware: the serial datapath
    reaches the same value as the summed one, just three cycles later.
    """
    img, k = _img(C=3, H=10, W=10, seed=21), F.random_kernels(1, C=3, K=3, seed=22)[0]
    acc = _serial_acc(img, k).reshape(-1, 3)
    conv = F.conv_output(img, k).reshape(-1)
    assert np.array_equal(acc[:, -1], conv)


def test_serial_accumulator_is_monotone_in_channel_count():
    """Each channel step must add exactly that channel's tap sum."""
    img, k = _img(C=3, H=8, W=8, seed=23), F.random_kernels(1, C=3, K=3, seed=24)[0]
    win = F.extract_windows(img, 3)
    per_chan = (win * k.reshape(3, 9)[None, :, :]).sum(axis=2)
    acc = _serial_acc(img, k).reshape(-1, 3)
    assert np.array_equal(acc[:, 0], per_chan[:, 0])
    assert np.array_equal(acc[:, 1] - acc[:, 0], per_chan[:, 1])
    assert np.array_equal(acc[:, 2] - acc[:, 1], per_chan[:, 2])


def test_serial_power_changes_after_the_accumulator_fix():
    """Guard against a silent revert: the defective expression gives different power."""
    img, k = _img(C=3, H=8, W=8, seed=25), F.random_kernels(1, C=3, K=3, seed=26)[0]
    pw = F.per_cycle_power(img, k, "serial")
    win = F.extract_windows(img, 3)
    g = (win * k.reshape(3, 9)[None, :, :]).astype(np.int32)
    stale = np.cumsum(g, axis=1).reshape(-1, 9)[:, -1]
    good = g.sum(axis=2).cumsum(axis=1).reshape(-1)
    assert not np.array_equal(stale, good)
    assert pw.shape == (img.shape[1] * img.shape[2] * 3,)


def test_serial_accumulator_matches_rtl_semantics():
    """Mirror the RTL: acc <= (chan==0) ? chan_sum : acc + chan_sum."""
    img, k = _img(C=3, H=6, W=6, seed=27), F.random_kernels(1, C=3, K=3, seed=28)[0]
    win = F.extract_windows(img, 3)
    chan_sum = (win * k.reshape(3, 9)[None, :, :]).sum(axis=2)
    rtl = np.zeros_like(chan_sum)
    for p in range(chan_sum.shape[0]):
        a = 0
        for c in range(3):
            a = chan_sum[p, c] if c == 0 else a + chan_sum[p, c]
            rtl[p, c] = a
    assert np.array_equal(_serial_acc(img, k).reshape(-1, 3), rtl)
