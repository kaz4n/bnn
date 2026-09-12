"""RGB first-layer convolution forward + power model (SIMULATION ONLY).

Phase 0 of the RGB input-recovery study. Nothing here touches hardware. This module
answers one question and no other:

    Does per-channel information survive the first-layer accumulation, at zero
    measurement noise, as a function of the accelerator's channel dataflow?

Why the dataflow is the whole question
--------------------------------------
The grayscale attacks (Wei et al. ACSAC'18; Huegle et al. FCCM 2023) both set input
channels to 1, so neither ever exercised channel accumulation. Extending them to RGB
runs into the fact that a first-layer convolution sums over input channels: for a 3x3
kernel, 27 input values produce one accumulator value. Whether the individual channel
contributions remain observable depends entirely on WHAT THE HARDWARE REGISTERS, and
that is an architecture choice, not a law:

  "serial"    Channels are iterated over cycles (C cycles per output pixel). Each
              channel's products land in the datapath in a cycle of their own, so the
              per-cycle power sequence carries per-channel structure directly. This is
              the common dataflow for resource-constrained FPGA accelerators that reuse
              one MAC array across channels.

  "parallel"  All C*K*K products are held in their own registers, feeding one adder
              tree. The product Hamming distances are per-channel quantities even
              though the accumulator is not, so channel information is present but
              mixed with the accumulator term.

  "summed"    Only the accumulator is registered; products are combinational. One
              scalar per cycle carries all C*K*K values. This is the worst case and the
              one that motivates the pessimistic reading of the gap.

Reporting all three is the point. A result that holds only under "serial" is a claim
about a class of accelerators, not about RGB inference in general, and must be stated
that way.

Power model
-----------
Same standard CMOS dynamic-power / Hamming-distance model as attack/trace_sim.py, so
Phase 0 numbers are comparable to the existing grayscale work:

    power[c] = sum_m HD(tc(g_m[c]), tc(g_m[c-1])) + HD(tc(acc[c]), tc(acc[c-1]))

with the product term omitted under "summed". Kernels are +-1 (binarized first layer),
matching the attacked design, so |g_m| = pixel value.

This is a NOISE-FREE model by default. Measurement effects (PDN smear, dilution,
jitter, finite sampling) are applied separately by rgb_channel.py so that architectural
loss and bench loss can never be confused for one another -- the separation that the
feasibility review flagged as unresolved question 2.
"""
from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

BITS = 16
_POP16 = np.array([bin(i).count("1") for i in range(1 << BITS)], dtype=np.uint8)

DATAFLOWS = ("serial", "parallel", "summed")


def _hd(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamming distance of two int arrays under BITS-wide two's-complement wrap."""
    return _POP16[((a ^ b) & 0xFFFF).astype(np.uint16)]


def extract_windows(img: np.ndarray, K: int) -> np.ndarray:
    """All K x K zero-padded windows of a CHW image, in row-major output order.

    img : (C, H, W) integer array
    returns : (H*W, C, K*K) int32, tap order (channel, row, col)

    Vectorized deliberately: the grayscale helper in attack/common.py loops in Python
    over 784 cycles, which is tolerable at 28x28 but not at 128x128 (16384 cycles) for
    thousands of images.
    """
    if img.ndim != 3:
        raise ValueError(f"expected (C,H,W), got shape {img.shape}")
    C, H, W = img.shape
    p = K // 2
    padded = np.pad(img.astype(np.int32), ((0, 0), (p, p), (p, p)))
    # (C, H, W, K, K) -> (H, W, C, K, K) -> (H*W, C, K*K)
    win = sliding_window_view(padded, (K, K), axis=(1, 2))
    win = np.transpose(win, (1, 2, 0, 3, 4))
    return np.ascontiguousarray(win.reshape(H * W, C, K * K))


def per_cycle_power(img: np.ndarray, kernel: np.ndarray, dataflow: str = "serial") -> np.ndarray:
    """Noise-free per-cycle power for one image and one multi-channel kernel.

    img     : (C, H, W) uint8/int, pixel values 0..255
    kernel  : (C, K, K) in {-1, +1}
    returns : (n_cycles,) float64.  n_cycles = H*W*C for "serial", H*W otherwise.

    The returned length differing by dataflow is intentional and load-bearing: the
    serial design genuinely spends C times as many cycles in the first layer, and that
    extra observation budget is precisely the thing that makes channels separable. An
    honest comparison across dataflows must compare at equal TRACE LENGTH IN CYCLES,
    not at equal output-pixel count.
    """
    if dataflow not in DATAFLOWS:
        raise ValueError(f"dataflow must be one of {DATAFLOWS}, got {dataflow!r}")
    C, K, _ = kernel.shape
    if img.shape[0] != C:
        raise ValueError(f"kernel has {C} channels, image has {img.shape[0]}")

    win = extract_windows(img, K)                      # (NPIX, C, KK)
    kf = kernel.reshape(C, K * K).astype(np.int32)     # (C, KK)
    g = (win * kf[None, :, :]).astype(np.int32)        # (NPIX, C, KK) products

    if dataflow == "serial":
        # One cycle per (output pixel, channel). Within a cycle only that channel's
        # products are live; the accumulator carries the running sum over channels so
        # far, resetting at each new output pixel.
        npix = g.shape[0]
        gs = g.reshape(npix * C, K * K)                       # cycle-major
        # Sum the TAP axis first, then accumulate across channels. The earlier version
        # was `np.cumsum(g, axis=1).reshape(npix * C, K * K)[:, -1]`, which accumulated
        # channels but then kept only the LAST TAP of each cycle instead of the tap sum
        # -- for products 1..27 it produced [9, 27, 54] where the stated model gives
        # [45, 171, 378]. The RTL (rgb_linebuf_core.sv) was always correct:
        # `acc <= (chan == 0) ? chan_sum[chan] : acc + chan_sum[chan]`, where chan_sum is
        # the nine-tap sum. So this was a simulator-only defect; measured hardware
        # results are unaffected. Found in external review, 12 Sep 2026.
        acc = g.sum(axis=2).cumsum(axis=1).reshape(npix * C)
        gp = np.zeros_like(gs); gp[1:] = gs[:-1]
        accp = np.zeros_like(acc); accp[1:] = acc[:-1]
        return (_hd(gs, gp).sum(axis=1) + _hd(acc, accp)).astype(np.float64)

    # parallel / summed: one cycle per output pixel, accumulator over all channels
    gflat = g.reshape(g.shape[0], C * K * K)
    acc = gflat.sum(axis=1).astype(np.int32)
    accp = np.zeros_like(acc); accp[1:] = acc[:-1]
    pw = _hd(acc, accp).astype(np.float64)
    if dataflow == "parallel":
        gp = np.zeros_like(gflat); gp[1:] = gflat[:-1]
        pw = pw + _hd(gflat, gp).sum(axis=1)
    return pw


def power_features(img: np.ndarray, kernels: np.ndarray, dataflow: str = "serial") -> np.ndarray:
    """Per-cycle power for a stack of kernels.

    kernels : (n_kernels, C, K, K) in {-1,+1}
    returns : (n_kernels, n_cycles) float64

    Stacking across kernels is what the template attack in Wei et al. S7 consumes as its
    power feature vector; the generative route in Huegle et al. uses a single deployed
    kernel. Both are supported by varying n_kernels.
    """
    return np.stack([per_cycle_power(img, k, dataflow) for k in kernels])


def conv_output(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Reference 'same'-padded convolution output, summed over input channels.

    Present so the power model can be checked against the arithmetic it claims to
    describe: a dataflow change must never alter this value.
    """
    C, K, _ = kernel.shape
    win = extract_windows(img, K)                      # (NPIX, C, KK)
    kf = kernel.reshape(C, K * K).astype(np.int32)
    out = (win * kf[None, :, :]).sum(axis=(1, 2))
    H, W = img.shape[1], img.shape[2]
    return out.reshape(H, W)


def random_kernels(n: int, C: int = 3, K: int = 3, seed: int = 0) -> np.ndarray:
    """Binarized +-1 kernels, matching the first layer of the attacked BNN."""
    rng = np.random.default_rng(seed)
    return rng.choice(np.array([-1, 1], dtype=np.int32), size=(n, C, K, K))
