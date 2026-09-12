"""Phase 1: bench model calibrated against real CW305/CW-Lite captures.

Still simulation -- but the noise level is no longer invented. It is measured from a
capture on this bench, so an RGB prediction made through this model inherits real
measured statistics instead of a guess.

Why this exists
---------------
Phase 0 established an architectural ceiling at zero noise. It could not say whether
this bench can reach it, because the feasibility review's unresolved question 2 noted
that the CW-Lite's 4 samples/cycle is a limit independent of colour. This module
separates the two by measuring the bench's actual signal-to-residual ratio and then
applying exactly that to the RGB probes.

The calibration anchor
----------------------
`host/traces_gfash_noamp_train60k` is the ideal reference: a real capture
(`capture_mode: live_cw305_chipwhisperer`) of a ZERO-amplifier grey design, single-shot
(`avg: 1`), on which the grayscale Power2Picture-style attack demonstrably works
(MSSIM 0.724, recognition 0.746 on 2000 held-out images). So its measured noise level is
known to be SUFFICIENT for grayscale recovery. That makes it a calibration point with a
known outcome attached, not just a number.

Noise model
-----------
Writing the per-window measured feature as

    feature = a * model_power + residual,     residual ~ N(0, sigma_n)

the Pearson correlation r between feature and model (after removing the fixed
per-window template, which is data-independent and cancels) fixes the ratio

    k = sigma_n / (a * sigma_model) = sqrt(1/r^2 - 1)

k is the only free parameter and it is measured, not chosen. Phase 0's `noise` argument
is in the same units (multiples of the per-image power standard deviation), so a
calibrated k can be passed straight into `rgb_probe.recovery_probe(noise=k)`.

Averaging divides sigma_n by sqrt(N), so k_effective = k / sqrt(N). Averaging is the
lever both Wei et al. (S7) and Moini et al. rely on, and Power2Picture S2 criticises the
latter for needing >1000x. It is therefore reported as a sweep, not assumed.

What this model does NOT claim
------------------------------
It is a scalar-SNR model with an optional smear kernel. It does not reproduce PDN
transfer functions, thermal drift, trigger jitter or the amplifier's frequency response.
It predicts how much per-channel information a read-out can extract at a measured noise
level -- not what a specific bitstream will produce. Phase 2 measures that.
"""
from __future__ import annotations

import glob
import json
import os

import numpy as np

POP16 = np.array([bin(i).count("1") for i in range(1 << 16)], dtype=np.uint8)

# Featurization windows discovered by measurement (see calibrate_from_capture).
# Signal is concentrated in the settling region a few cycles into each dwell period;
# averaging the whole dwell dilutes it.
FEATURE_WINDOWS = {
    "mean_all": None,          # whole dwell -- what the existing pipeline uses
    "settle_6_13": (6, 14),    # measured best on the reference capture
    "settle_8_11": (8, 12),
}


def _hd(a, b):
    return POP16[((a ^ b) & 0xFFFF).astype(np.uint16)]


def grayscale_model_power(img: np.ndarray, kernel: np.ndarray, out_side: int) -> np.ndarray:
    """Noise-free per-window power for the deployed grayscale design.

    'valid' 3x3 convolution in row-major scan order, matching the CW305 leakage core
    (n_windows = out_side^2). Used only for calibration, so it must match the capture
    geometry rather than the same-padded geometry in attack/common.py.
    """
    K = kernel.shape[0]
    W = np.lib.stride_tricks.sliding_window_view(img.astype(np.int32), (K, K))
    W = W.reshape(out_side * out_side, K * K)
    g = (W * kernel.reshape(-1)[None, :].astype(np.int32)).astype(np.int32)
    acc = g.sum(1).astype(np.int32)
    gp = np.zeros_like(g); gp[1:] = g[:-1]
    ap = np.zeros_like(acc); ap[1:] = acc[:-1]
    return (_hd(g, gp).sum(1) + _hd(acc, ap)).astype(np.float64)


def calibrate_from_capture(trace_dir: str, n_traces: int = 1024,
                           feature: str = "settle_6_13") -> dict:
    """Measure this bench's signal-to-residual ratio k from a real capture.

    Returns the measured correlation, the implied k, the same for every featurization in
    FEATURE_WINDOWS, and enough provenance to tie the number to a specific bitstream.

    Reads only. Touches no hardware.
    """
    man_path = os.path.join(trace_dir, "capture_manifest.json")
    man = json.load(open(man_path))
    shards = sorted(glob.glob(os.path.join(trace_dir, "shard*.npz")))
    if not shards:
        raise SystemExit(f"no shard*.npz in {trace_dir}")

    z = np.load(shards[0])
    scale = float(man.get("trace_scale", 32767.0))
    tr = z["traces"][:n_traces].astype(np.float64) / scale
    imgs = z["images"][:n_traces].astype(np.int32)

    kern = np.load(man["kernels"].replace("\\", "/"))[man["kernel_index"]].astype(np.int32)
    spw, nw = man["samples_per_window"], man["n_windows"]
    us, out_side = man["useful_samples"], man["out_side"]

    model = np.stack([grayscale_model_power(im, kern, out_side) for im in imgs])
    mc = model - model.mean(0)                       # drop fixed per-window template
    samp = np.abs(tr[:, :us].reshape(len(tr), nw, spw))
    sc = samp - samp.mean(0)

    per_feature = {}
    for name, win in FEATURE_WINDOWS.items():
        f = sc.mean(2) if win is None else sc[:, :, win[0]:win[1]].mean(2)
        r = float(np.corrcoef(f.ravel(), mc.ravel())[0, 1])
        per_feature[name] = {
            "correlation": r,
            "k_noise_over_signal": float(np.sqrt(max(1.0 / (r * r) - 1.0, 0.0))) if r else None,
        }

    # Distinguish genuine inter-window smear from the model's own autocorrelation: the
    # conv window overlaps between adjacent outputs, so neighbouring model values are
    # correlated even with a perfect measurement.
    f_best = sc.mean(2) if FEATURE_WINDOWS[feature] is None else \
        sc[:, :, FEATURE_WINDOWS[feature][0]:FEATURE_WINDOWS[feature][1]].mean(2)
    r0 = float(np.corrcoef(f_best.ravel(), mc.ravel())[0, 1])
    r1 = float(np.corrcoef(f_best[:, :-1].ravel(), mc[:, 1:].ravel())[0, 1])
    m1 = float(np.corrcoef(mc[:, :-1].ravel(), mc[:, 1:].ravel())[0, 1])
    smear_excess = r1 - r0 * m1     # >0 means measured leakage beyond window overlap

    return {
        "trace_dir": trace_dir,
        "bitstream_sha256": man.get("bitstream_sha256"),
        "capture_mode": man.get("capture_mode"),
        "input_mode": man.get("input_mode"),
        "avg": man.get("avg"),
        "samples_per_cycle": man.get("samples_per_cycle"),
        "dwell": man.get("dwell"),
        "gain_db": man.get("gain_db"),
        "fpga_freq_hz": man.get("fpga_freq_hz"),
        "n_traces_used": int(len(tr)),
        "per_feature": per_feature,
        "chosen_feature": feature,
        "k_calibrated": per_feature[feature]["k_noise_over_signal"],
        "smear": {
            "measured_lag1_corr": r1,
            "model_lag1_autocorr": m1,
            "predicted_lag1_if_no_smear": r0 * m1,
            "excess_attributable_to_smear": smear_excess,
        },
        "note": ("k is sigma_noise / sigma_signal in per-window centred feature units; "
                 "the grayscale attack works at this k, which makes it a sufficiency "
                 "anchor, not merely a noise level"),
    }


def k_after_averaging(k: float, n_avg: int) -> float:
    """Effective k after averaging n_avg captures. Noise falls as sqrt(n)."""
    if n_avg < 1:
        raise ValueError("n_avg must be >= 1")
    return float(k) / float(np.sqrt(n_avg))


def smear_cycles(power: np.ndarray, tau: float) -> np.ndarray:
    """One-pole leaky integrator across cycles, tau in cycles. tau<=0 is a no-op.

    Applied along the LAST axis (cycles). This is the operation that threatens a
    channel-serial dataflow specifically: serial channels occupy adjacent cycles, so
    smear mixes exactly the quantities the attack needs to separate. A dwell period per
    channel is the design response, and its effect is what Phase 1 measures.
    """
    if tau <= 0:
        return power
    a = np.exp(-1.0 / tau)
    out = np.empty_like(power, dtype=np.float64)
    acc = np.zeros(power.shape[:-1], dtype=np.float64)
    for n in range(power.shape[-1]):
        acc = (1.0 - a) * power[..., n] + a * acc
        out[..., n] = acc
    return out


def apply_dwell(power: np.ndarray, dwell: int) -> np.ndarray:
    """Hold each cycle's value for `dwell` cycles, as the CW305 core does.

    Dwell is why smear does not destroy the reference capture: each window gets settling
    time, and the measured intra-window profile shows activity confined to a few samples
    early in the period. Repeating before smearing reproduces that.
    """
    if dwell <= 1:
        return power
    return np.repeat(power, dwell, axis=-1)
