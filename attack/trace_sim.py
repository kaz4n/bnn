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

# MUST match hls/bnn_conv1.cpp. The hardware amplifier is FANOUT kernel-dependent MAC
# lanes; lane f computes products of the REAL kernel with the pixel XOR'd by c_f = f*LANE_C
# (mod 256), c_0 = 0. This models the SAME leakage the bitstream produces, so the attack
# validated here is the attack that runs on the bench (see the header of bnn_conv1.cpp).
FANOUT = 16
LANE_C = 0x35

# 16-bit population-count table -> vectorized Hamming distance of the two's-complement
# datapath state, matching the int16 product/accumulator registers pg/pacc in the HLS.
_POP16 = np.array([bin(i).count("1") for i in range(1 << BITS)], dtype=np.uint16)


def _tc(v):
    return int(v) & (2**BITS - 1)            # two's-complement wrap to BITS


def _all_windows(img, K):
    """[NCYC, K*K] window pixel values in the same tap order as common.window_pixels."""
    W = np.zeros((NCYC, K * K), dtype=np.int32)
    for c in range(NCYC):
        y, x = cyc_to_yx(c)
        W[c] = window_pixels(img, y, x, K)
    return W


def per_cycle_power(img, kernel, fanout=FANOUT):
    """Length-784 per-cycle power (noise-free) for one image, one KxK kernel.

    Models the HLS FANOUT amplifier: power[c] = sum over lanes f of
      sum_m HD(pg_f[m][c], pg_f[m][c-1]) + HD(pacc_f[c], pacc_f[c-1])
    where pg_f[m] = kern[m]*(pixel[m] ^ c_f) and pacc_f = sum_m pg_f[m], c_f = f*LANE_C.
    The kernel sign is applied identically in every lane, so the across-kernel variance
    (the S7 template signal) is amplified ~fanout x rather than cancelled.
    """
    K = kernel.shape[0]
    kf = kernel.flatten().astype(np.int32)              # +-1 weights
    W = _all_windows(img, K)                             # [NCYC, KK] pixel values 0..255
    pw = np.zeros(NCYC, dtype=np.float64)
    for f in range(fanout):
        cf = (f * LANE_C) & 0xFF
        g = (kf * (W ^ cf)).astype(np.int32)            # products +-(pixel ^ c_f)
        acc = g.sum(axis=1).astype(np.int32)            # per-cycle accumulator
        gp = np.zeros_like(g); gp[1:] = g[:-1]          # previous-cycle products (c-1)
        accp = np.zeros_like(acc); accp[1:] = acc[:-1]
        hd_g = _POP16[((g ^ gp) & 0xFFFF).astype(np.uint16)].sum(axis=1)
        hd_a = _POP16[((acc ^ accp) & 0xFFFF).astype(np.uint16)]
        pw += hd_g + hd_a
    return pw


# Fixed data-INDEPENDENT global-activity template (clock tree / USB / unused fabric), one
# value per (kernel, cycle). Reused across images so it models a systematic, NOT random,
# contaminant -- the thing averaging cannot remove. Sized lazily to the kernel count.
_GLOBAL_RNG = np.random.default_rng(0xC0FFEE)
_GLOBAL_CACHE = {}


def _global_template(n_kernels):
    # The data-independent global activity (clock tree / USB / unused fabric) is the SAME
    # regardless of which kernel is loaded (the kernel is just register data), so the
    # template must be COMMON across kernels -- one per-cycle vector broadcast to every
    # kernel row. (Per-trace variability is injected separately as dilution_jitter, drawn
    # independently per capture.) A per-kernel-distinct template would fabricate spurious
    # across-kernel structure and mislead the S7 analysis.
    if n_kernels not in _GLOBAL_CACHE:
        base = np.abs(_GLOBAL_RNG.normal(0, 1, size=NCYC))
        _GLOBAL_CACHE[n_kernels] = np.tile(base, (n_kernels, 1))
    return _GLOBAL_CACHE[n_kernels]


def simulate_image(img, kernels, samples_per_cycle=4, noise=0.5, gain=1.0, seed=None,
                   smear_tau=0.0, dilution=0.0, dilution_jitter=0.0):
    """
    kernels: (n_kernels, K, K) in {-1,+1}.
    Returns raw trace [n_kernels, n_cycles*samples_per_cycle] mimicking CW synchronous
    capture (a fixed pulse per cycle scaled by per-cycle power + Gaussian noise).

    HARDWARE-REGIME KNOBS (default 0 == the original clean sim; leaving them 0 keeps every
    prior result reproducible). Set them to model why REAL CW305 capture fails where this
    sim succeeds (see memory dilution-rootcause; scratchpad/sim_confound*.py):
      smear_tau        inter-cycle PDN smear: one-pole leaky integrator, tau in CYCLES.
                       Secondary effect (corr 0.8->~0.36 at tau=10); does NOT alone reach
                       the measured ~0.1.
      dilution         data-independent global-activity term, in multiples of the per-cycle
                       signal std. THIS reproduces the collapse: dilution=10 -> corr ~0.06.
                       Real 0.57mW conv in a big A100T is ~100-1000x.
      dilution_jitter  per-trace fractional gain variation of the global term (e.g. 0.02 =
                       2%). This VARIABILITY is the true floor: the mean cancels under
                       common-mode subtraction, the jitter does not (5% -> corr ~0.09).
    """
    rng = np.random.default_rng(seed)
    spc = samples_per_cycle
    pulse = np.hanning(spc + 2)[1:-1]; pulse = pulse / pulse.sum()   # per-cycle shape
    traces = np.zeros((len(kernels), NCYC * spc), dtype=np.float32)
    sig_std = None
    if dilution > 0:                       # scale global term relative to signal magnitude
        sig_std = np.std([per_cycle_power(img, k) for k in kernels]) or 1.0
        gtmpl = _global_template(len(kernels))
    for ki, k in enumerate(kernels):
        pw = per_cycle_power(img, k) * gain
        sig = np.repeat(pw, spc) * np.tile(pulse, NCYC)
        row = sig + rng.normal(0, noise, size=sig.size)
        if smear_tau > 0:                  # continuous-time PDN low-pass mixes cycles
            a = np.exp(-1.0 / (smear_tau * spc))
            row = signal_lfilter_1pole(row, a)
        if dilution > 0:                   # add correlated data-independent global activity
            g = np.repeat(gtmpl[ki], spc)
            jit = 1.0 + (rng.normal(0, dilution_jitter) if dilution_jitter > 0 else 0.0)
            row = row + dilution * sig_std * g * jit
        traces[ki] = row
    return traces


def signal_lfilter_1pole(x, a):
    """y[n] = (1-a) x[n] + a y[n-1]  -- dependency-free one-pole IIR (no scipy needed)."""
    y = np.empty_like(x, dtype=np.float64)
    acc = 0.0
    one_minus_a = 1.0 - a
    for n in range(x.size):
        acc = one_minus_a * x[n] + a * acc
        y[n] = acc
    return y


def save_npz(path, img, label, kernels, **kw):
    traces = simulate_image(img, kernels, **kw)
    np.savez(path, traces=traces, kernel_ids=np.arange(len(kernels)),
             image=img.astype(np.uint8), label=int(label),
             samples_per_cycle=kw.get("samples_per_cycle", 4), n_cycles=NCYC)
    return traces
