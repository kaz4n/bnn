"""
P3 -- power-trace SIMULATOR for the layer-1 line-buffer conv unit.

Lets the full attack run end-to-end WITHOUT the CW305/CW-Lite bench. Emits the SAME
.npz format as host/cw305_bnn_capture.py, so swapping to real traces later needs no
change to the attack code.

Power model (standard CMOS dynamic-power / Hamming-distance model used in DPA):
dynamic power per cycle ~ switching activity. At cycle c the K*K multipliers hold
products g_ab = kernel_ab * pixel_ab (kernel in {-1,+1}, so |g|=pixel), and the adder
tree holds acc = sum g. Activity from cycle c-1 to c is the Hamming distance of the
two's-complement datapath state:

    power[c] = sum_ab HD(tc(g_ab[c]), tc(g_ab[c-1])) + HD(tc(acc[c]), tc(acc[c-1]))

This reproduces the two effects the attacks exploit:
  * background (window unchanged between cycles) -> products unchanged -> HD~=0 -> low
    power  (paper S6.1 background detection),
  * power depends jointly on the related pixels and the kernel, so the across-kernel
    power-feature-vector encodes the pixels (paper S7 power template).

Real CW traces carry the same information with measurement noise; this sim adds Gaussian
noise + a per-cycle pulse shape so the P3 front-end (power_extract.py) is exercised.
"""
import numpy as np
from common import LINE, NCYC, cyc_to_yx, window_pixels

BITS = 16


def _tc(v):
    return int(v) & (2**BITS - 1)            # two's-complement wrap to BITS


def _hd(a, b):
    return bin(_tc(a) ^ _tc(b)).count("1")


def per_cycle_power(img, kernel):
    """Return length-784 per-cycle power (noise-free) for one image, one KxK kernel."""
    K = kernel.shape[0]
    kflat = kernel.flatten().astype(np.int32)
    pw = np.zeros(NCYC, dtype=np.float64)
    prev_g = np.zeros(K * K, dtype=np.int32)
    prev_acc = 0
    for c in range(NCYC):
        y, x = cyc_to_yx(c)
        px = window_pixels(img, y, x, K)
        g = kflat * px                        # products (kernel +-1)
        acc = int(g.sum())
        hd = sum(_hd(g[i], prev_g[i]) for i in range(K * K)) + _hd(acc, prev_acc)
        pw[c] = hd
        prev_g, prev_acc = g, acc
    return pw


def simulate_image(img, kernels, samples_per_cycle=4, noise=0.5, gain=1.0, seed=None):
    """
    kernels: (n_kernels, K, K) in {-1,+1}.
    Returns raw trace [n_kernels, n_cycles*samples_per_cycle] mimicking CW synchronous
    capture (a fixed pulse per cycle scaled by per-cycle power + Gaussian noise).
    """
    rng = np.random.default_rng(seed)
    spc = samples_per_cycle
    pulse = np.hanning(spc + 2)[1:-1]; pulse = pulse / pulse.sum()   # per-cycle shape
    traces = np.zeros((len(kernels), NCYC * spc), dtype=np.float32)
    for ki, k in enumerate(kernels):
        pw = per_cycle_power(img, k) * gain
        sig = np.repeat(pw, spc) * np.tile(pulse, NCYC)
        traces[ki] = sig + rng.normal(0, noise, size=sig.size)
    return traces


def save_npz(path, img, label, kernels, **kw):
    traces = simulate_image(img, kernels, **kw)
    np.savez(path, traces=traces, kernel_ids=np.arange(len(kernels)),
             image=img.astype(np.uint8), label=int(label),
             samples_per_cycle=kw.get("samples_per_cycle", 4), n_cycles=NCYC)
    return traces
