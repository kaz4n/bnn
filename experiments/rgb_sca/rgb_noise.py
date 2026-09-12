"""Sample-granularity bench model built from REAL measured residuals.

Replaces the scalar-per-window, i.i.d.-Gaussian noise model used in Phase 1, which was
measurably pessimistic: replaying a capture whose outcome is known gave MSSIM 0.584
against a measured 0.724, and 2.8x more training data closed only 0.040 of that gap, so
the shortfall was systematic rather than a training-budget artifact.

WHY THE OLD MODEL WAS WRONG
---------------------------
Measured on `traces_gfash_noamp_train60k`, the residual left after removing the
data-independent template and the model-power term is strongly CORRELATED, not white:

    across windows, same sample position   lag-1 +0.59,  lag-4 +0.72
    across samples within a window         lag-1 +0.21
    per-trace common mode                  0.175 of total residual std

White noise is the hardest case for a learned reconstructor -- a CNN can learn to reject
structure it cannot reject from white noise. Modelling correlated bench noise as i.i.d.
Gaussian therefore understates achievable reconstruction, which is exactly the direction
of the observed gap.

THE FIX: DON'T MODEL THE NOISE, REUSE IT
----------------------------------------
Rather than fit an AR process to that structure and hope the fit is faithful, this module
decomposes a real capture

    |trace|[image, window, sample] = template[window, sample]
                                   + gain[sample] * power[image, window]
                                   + residual[image, window, sample]

and then synthesizes new traces by keeping the measured `template`, `gain` and `residual`
and substituting a NEW power term computed from whatever image is being simulated. The
noise in the result is real bench noise, with every correlation it actually has, because
it was never modelled in the first place.

WHAT THIS STILL DOES NOT CAPTURE
--------------------------------
The residual pool comes from a grey 28x28 design. Simulating a different geometry (RGB
32x32) tiles the pool across windows, so long-range window correlation is periodic at the
pool width and a seam exists at the wrap. Local structure, which dominates, is preserved.
Any residual that is genuinely a function of the image would be inherited from the wrong
image -- but such a component would by definition not be residual, so this is second
order. The decomposition is linear in power; a strongly nonlinear leakage term would be
absorbed into the residual and carried over unchanged.
"""
from __future__ import annotations

import glob
import json
import os

import numpy as np

from . import rgb_bench as B


def characterize_capture(trace_dir: str, n_traces: int = 1500) -> dict:
    """Decompose a real capture into template, per-sample gain and a residual pool.

    Read-only. Touches no hardware.

    Returns a dict with:
      template  (n_windows, samples_per_window)  data-independent, cancels in the real
                                                 measurement but must be present so the
                                                 featurizer sees a realistic signal
      gain      (samples_per_window,)            regression of centred measurement on
                                                 centred model power, per sample position;
                                                 concentrated in the settling region
      residual  (n_traces, n_windows, spw)       the real noise, with all its correlation
      plus the geometry needed to synthesize matching traces.
    """
    man = json.load(open(os.path.join(trace_dir, "capture_manifest.json")))
    shards = sorted(glob.glob(os.path.join(trace_dir, "shard*.npz")))
    if not shards:
        raise SystemExit(f"no shard*.npz in {trace_dir}")

    scale = float(man.get("trace_scale", 32767.0))
    spw, nw = man["samples_per_window"], man["n_windows"]
    us, out_side = man["useful_samples"], man["out_side"]
    kern = np.load(man["kernels"].replace("\\", "/"))[man["kernel_index"]].astype(np.int32)

    tr, imgs, taken = [], [], 0
    for p in shards:
        z = np.load(p)
        take = min(len(z["traces"]), n_traces - taken)
        tr.append(z["traces"][:take].astype(np.float64) / scale)
        imgs.append(z["images"][:take].astype(np.int32))
        taken += take
        if taken >= n_traces:
            break
    tr = np.concatenate(tr)
    imgs = np.concatenate(imgs)

    power = np.stack([B.grayscale_model_power(im, kern, out_side) for im in imgs])
    samp = np.abs(tr[:, :us].reshape(len(tr), nw, spw))

    template = samp.mean(0)
    sc = samp - template
    pc = power - power.mean(0)
    denom = float((pc * pc).sum()) or 1.0
    gain = np.array([float((sc[:, :, s] * pc).sum()) / denom for s in range(spw)])
    residual = (sc - gain[None, None, :] * pc[:, :, None]).astype(np.float32)

    return {
        "trace_dir": trace_dir,
        "bitstream_sha256": man.get("bitstream_sha256"),
        "capture_mode": man.get("capture_mode"),
        "avg": man.get("avg"),
        "samples_per_window": int(spw),
        "n_windows_pool": int(nw),
        "out_side": int(out_side),
        "samples_per_cycle": int(man["samples_per_cycle"]),
        "dwell": int(man["dwell"]),
        "n_traces": int(len(tr)),
        "template": template.astype(np.float32),
        "gain": gain.astype(np.float32),
        "residual": residual,
        "power_std": float(pc.std()),
    }


def synthesize_samples(power: np.ndarray, ch: dict, n_avg: int = 1,
                       seed: int = 0, precentred: bool = False) -> np.ndarray:
    """Build sample-granularity traces for arbitrary per-window model power.

    power : (n_images, n_windows) noise-free model power for the images being simulated
    n_avg : averaging factor, implemented by averaging N independent draws from the
            residual pool.

            CAVEAT -- simulated averaging is probably OPTIMISTIC. Pool draws are fully
            independent, so this model delivers the full sqrt(N) benefit (round-trip:
            corr 0.34 -> 0.76 at N=10 -> 0.94 at N=50). The project's own RO-channel
            measurements found correlation PLATEAUS with averaging on real hardware
            (0.28@50 -> 0.31@200 -> 0.33@800), which is the signature of a persistent
            component that averaging cannot remove. Any residual structure that repeats
            across captures of the same image is, in this pool, spread across different
            images and therefore averages away when on the bench it would not.

            So treat avg=1 as validated (it round-trips to the measured correlation) and
            any avg>1 number as an UPPER bound on what averaging buys.

    returns (n_images, n_windows * samples_per_window) float32, in the same units and
    layout the capture script writes, so `rgb_generator.featurize_rgb` applies unchanged.
    """
    rng = np.random.default_rng(seed)
    n_img, nw = power.shape
    spw = ch["samples_per_window"]
    pool, pool_nw = ch["residual"], ch["n_windows_pool"]

    # Tile the residual pool across windows when the simulated geometry is wider than the
    # captured one. Local correlation is preserved; see the module docstring.
    widx = np.arange(nw) % pool_nw

    pc = power if precentred else power - power.mean(axis=0, keepdims=True)
    tmpl = ch["template"][widx, :]                       # (nw, spw)
    sig = pc[:, :, None] * ch["gain"][None, None, :]     # (n_img, nw, spw)

    noise = np.zeros((n_img, nw, spw), dtype=np.float32)
    for _ in range(max(1, n_avg)):
        rows = rng.integers(0, len(pool), size=n_img)
        noise += pool[rows][:, widx, :]
    noise /= float(max(1, n_avg))

    return (tmpl[None, :, :] + sig + noise).reshape(n_img, nw * spw).astype(np.float32)


def simulate_features_sampled(images, kernels, dataflow, ch, n_avg=1, seed=0,
                              feature="settle", chunk=None, power=None):
    """End-to-end: images -> model power -> sample-granularity trace -> features.

    Goes through exactly the featurizer the real pipeline uses, so a simulated result and
    a hardware result are processed identically.

    Synthesis is CHUNKED. The full sample-granularity array is n_images x n_windows x
    samples_per_window, which at 56,000 images is 4.5 GiB -- it does not fit, and it does
    not need to: only the featurized result (a factor of samples_per_window smaller) is
    kept. Centring of the power term is done once over ALL images before chunking, so the
    chunk boundary does not change the result.
    """
    from . import rgb_forward as F
    from . import rgb_generator as G

    # GEOMETRY. `power` may be supplied by the caller, and must be when the simulated
    # geometry has to match the capture's. rgb_forward.power_features uses 'same'
    # padding (IMG_SIDE^2 windows) whereas the grey capture is a 'valid' convolution
    # (out_side^2 = 676). Feeding same-padded power against a valid-geometry residual
    # pool silently tiles 784 windows over a 676-window pool and compares against the
    # wrong window count -- so the anchor passes its own valid-geometry power in.
    if power is None:
        power = np.stack([F.power_features(im, kernels, dataflow).ravel()
                          for im in images])
    power = np.asarray(power, dtype=np.float64)
    # Centre once, globally; synthesize_samples would otherwise centre per chunk and make
    # the output depend on how the work happened to be divided.
    power = power - power.mean(axis=0, keepdims=True)
    man = {"samples_per_cycle": ch["samples_per_cycle"], "dwell": ch["dwell"]}

    # Size the chunk by the actual synthesis footprint, not a fixed image count. The
    # intermediate is n_chunk x n_windows x samples_per_window float64; the serial
    # dataflow has 3x the windows of summed, so a fixed 4000-image chunk that fits one
    # overflows the other (observed: 2.24 GiB for 4000 x 2352 x 32).
    if chunk is None:
        per_img_bytes = power.shape[1] * ch["samples_per_window"] * 8
        chunk = max(64, min(4000, int(512e6 // max(per_img_bytes, 1))))

    out = None
    for start in range(0, len(power), chunk):
        block = power[start:start + chunk]
        traces = synthesize_samples(block, ch, n_avg, seed + start, precentred=True)
        feats = G.featurize_rgb(traces, man, feature)
        if out is None:
            out = np.empty((len(power),) + feats.shape[1:], dtype=feats.dtype)
        out[start:start + len(feats)] = feats
        del traces, feats
    return out
