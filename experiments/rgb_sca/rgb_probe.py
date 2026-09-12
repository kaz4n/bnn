"""Phase 0 probes: does per-channel information survive the first-layer accumulation?

SIMULATION ONLY. No hardware, no capture, no bench. Noise-free unless explicitly asked.

This module answers unresolved question 1 from the feasibility review (does any
per-channel information survive channel summing) BEFORE any reconstruction model is
trained, and separately from unresolved question 2 (is this bench able to extract it).
Keeping them separate is the whole point: a negative result from a trained generator on
noisy traces cannot distinguish "the architecture destroys color" from "the model was
too small" from "the bench is too coarse". The probes below can.

Two probes, in increasing strength of claim.

1. distinguisher()  -- EXACT, information-theoretic, no learning involved.
   Compares the power trace of an image against the trace of the same image with its
   colour channels permuted. If the two traces are bit-identical, channel identity is
   provably unobservable through this channel and the hypothesis is dead at the
   architecture level regardless of model, bench or dataset. If they differ, channel
   information is PRESENT -- which is necessary but not sufficient for recovery.

2. recovery_probe() -- LEARNED, but linear and cheap.
   Ridge regression from local power features to the centre pixel's (R,G,B), scored
   against two controls that the published literature does not report:
     * prior-only   : predict the training-set mean. Any honest claim of recovery must
                      beat this, and "beats the prior" is the actual claim being tested.
     * shuffled     : real traces, but paired with the WRONG images. Destroys the
                      input-trace association while preserving every marginal
                      distribution. Should collapse to prior-only; if it does not, the
                      pipeline is leaking through something other than the trace.

A linear probe is deliberately weak. It gives a LOWER bound on extractable information.
A negative linear result does not close the question; a positive one establishes the
information is there and linearly accessible, which is a strong and early answer.
"""
from __future__ import annotations

import numpy as np

from . import rgb_forward as F


# ------------------------------------------------------------------ probe 1: exact


def distinguisher(img: np.ndarray, kernels: np.ndarray, dataflow: str,
                  perm: tuple[int, ...] = (2, 1, 0)) -> dict:
    """Does permuting the colour channels change the power trace at all?

    perm defaults to (2,1,0): swap R and B, leave G. Any non-identity permutation works.

    Returns a dict with:
      identical      True if the permuted image yields a bit-identical trace
      frac_differing fraction of cycles whose power changed
      rel_l1         mean |dP| / mean P, a scale-free magnitude of the change

    Interpretation, stated carefully:
      identical == True  -> channel identity is unobservable for THIS kernel set and
                            dataflow. Decisive negative.
      identical == False -> channel information is present in principle. NOT a claim
                            that it is recoverable under noise.
    """
    permuted = img[list(perm)]
    a = F.power_features(img, kernels, dataflow)
    b = F.power_features(permuted, kernels, dataflow)
    d = np.abs(a - b)
    denom = np.mean(a) or 1.0
    return {
        "dataflow": dataflow,
        "identical": bool(np.array_equal(a, b)),
        "frac_differing": float(np.mean(d > 0)),
        "rel_l1": float(np.mean(d) / denom),
    }


def kernel_channel_asymmetry(kernels: np.ndarray) -> float:
    """Fraction of kernels whose per-channel weight planes are not all identical.

    Reported alongside distinguisher() because it is the mechanism behind it. A kernel
    with k[R] == k[G] == k[B] contributes nothing to channel identity in the accumulator
    (permuting channels leaves the sum unchanged), so a channel-symmetric kernel set can
    produce a false negative that says more about the probe kernels than the hardware.
    Trained first-layer kernels are generically asymmetric; one-hot probe kernels may not
    be.
    """
    asym = 0
    for k in kernels:
        planes = k.reshape(k.shape[0], -1)
        if not np.all(planes == planes[0]):
            asym += 1
    return float(asym) / len(kernels)


# ------------------------------------------------------------------ features


def pixel_features(power: np.ndarray, n_pixels: int, cycles_per_pixel: int,
                   context: int = 1) -> np.ndarray:
    """Per-output-pixel feature vectors from a (n_kernels, n_cycles) power array.

    For output pixel i, takes that pixel's own cycles plus `context` neighbouring
    pixels either side, across every kernel, flattened. Neighbour context matters
    because the line-buffer window overlaps between adjacent output pixels, so a pixel's
    value influences several cycles (this is the same overlap the template attack in Wei
    et al. S7.2 exploits via its "related pixels" union).

    returns : (n_pixels, n_kernels * cycles_per_pixel * (2*context+1))
    """
    n_k = power.shape[0]
    span = 2 * context + 1
    byp = power.reshape(n_k, n_pixels, cycles_per_pixel)
    idx = np.clip(np.arange(n_pixels)[:, None] + np.arange(-context, context + 1)[None, :],
                  0, n_pixels - 1)
    gathered = byp[:, idx, :]                       # (n_k, n_pixels, span, cpp)
    return np.transpose(gathered, (1, 0, 2, 3)).reshape(n_pixels, n_k * span * cycles_per_pixel)


def build_dataset(images: np.ndarray, kernels: np.ndarray, dataflow: str,
                  context: int = 1, noise: float = 0.0, seed: int = 0):
    """Assemble (features, targets) over a stack of images.

    images : (N, C, H, W) uint8
    returns X (N*H*W, D) float64, Y (N*H*W, C) float64 centre-pixel values 0..255,
            and img_id (N*H*W,) so shuffling can be done at IMAGE granularity.

    noise > 0 adds i.i.d. Gaussian noise scaled to the per-image power std. This is a
    crude stand-in only, used to check that a positive noise-free result degrades
    gracefully; it is NOT a bench model. Bench effects (PDN smear, dilution, jitter,
    finite sampling) belong to Phase 1 and are deliberately not modelled here.
    """
    rng = np.random.default_rng(seed)
    N, C, H, W = images.shape
    npix = H * W
    cpp = C if dataflow == "serial" else 1
    X, Y, ids = [], [], []
    for n in range(N):
        pw = F.power_features(images[n], kernels, dataflow)
        if noise > 0:
            pw = pw + rng.normal(0, noise * (np.std(pw) or 1.0), size=pw.shape)
        X.append(pixel_features(pw, npix, cpp, context))
        Y.append(images[n].reshape(C, npix).T.astype(np.float64))
        ids.append(np.full(npix, n))
    return np.concatenate(X), np.concatenate(Y), np.concatenate(ids)


# ------------------------------------------------------------------ probe 2: learned


def _ridge_fit(X: np.ndarray, Y: np.ndarray, alpha: float):
    """Closed-form ridge with an intercept. numpy only, no sklearn dependency."""
    mu_x, mu_y = X.mean(0), Y.mean(0)
    Xc, Yc = X - mu_x, Y - mu_y
    G = Xc.T @ Xc
    G.flat[:: G.shape[0] + 1] += alpha
    Wt = np.linalg.solve(G, Xc.T @ Yc)
    return Wt, mu_x, mu_y


def _ridge_predict(model, X):
    Wt, mu_x, mu_y = model
    return (X - mu_x) @ Wt + mu_y


def _scores(pred: np.ndarray, truth: np.ndarray) -> dict:
    """Per-channel MAE in 0..255 units, plus pooled. Never pool away the channels."""
    mae = np.abs(pred - truth).mean(0)
    return {
        "mae_per_channel": [float(v) for v in mae],
        "mae_pooled": float(mae.mean()),
    }


def recovery_probe(images: np.ndarray, kernels: np.ndarray, dataflow: str,
                   train_frac: float = 0.7, alpha: float = 1e3, context: int = 1,
                   noise: float = 0.0, seed: int = 0) -> dict:
    """Linear recovery of per-channel pixel values, against two controls.

    The split is by IMAGE, never by pixel: pixels from one image are highly correlated,
    so a pixel-level split leaks the test image into training and inflates every score.

    Returns per-channel MAE for the trace model, the prior-only baseline and the
    shuffled-trace control, plus the trace model's advantage over the prior.
    """
    N = len(images)
    rng = np.random.default_rng(seed)
    order = rng.permutation(N)
    n_tr = max(1, int(round(train_frac * N)))
    tr_imgs, te_imgs = set(order[:n_tr].tolist()), set(order[n_tr:].tolist())
    if not te_imgs:
        raise ValueError("need at least 2 images for a held-out split")

    X, Y, ids = build_dataset(images, kernels, dataflow, context, noise, seed)
    tr = np.isin(ids, list(tr_imgs))
    te = ~tr

    model = _ridge_fit(X[tr], Y[tr], alpha)
    trace_scores = _scores(_ridge_predict(model, X[te]), Y[te])

    # Control A -- prior only: the best constant predictor learned from training data.
    prior = np.repeat(Y[tr].mean(0)[None, :], te.sum(), axis=0)
    prior_scores = _scores(prior, Y[te])

    # Control B -- shuffled: real features, wrong images. Permute feature ROWS across
    # images while keeping targets in place, so every marginal is preserved and only the
    # association is destroyed.
    sh = X.copy()
    img_ids = np.unique(ids)
    perm = rng.permutation(len(img_ids))
    for src, dst in zip(img_ids, img_ids[perm]):
        sh[ids == dst] = X[ids == src]
    sh_model = _ridge_fit(sh[tr], Y[tr], alpha)
    shuffled_scores = _scores(_ridge_predict(sh_model, sh[te]), Y[te])

    adv = [p - t for p, t in zip(prior_scores["mae_per_channel"],
                                 trace_scores["mae_per_channel"])]
    return {
        "dataflow": dataflow,
        "n_images": int(N),
        "n_train_images": int(n_tr),
        "n_kernels": int(len(kernels)),
        "noise": float(noise),
        "trace": trace_scores,
        "prior_only": prior_scores,
        "shuffled": shuffled_scores,
        "advantage_over_prior_per_channel": [float(v) for v in adv],
        "advantage_over_prior_pooled": float(
            prior_scores["mae_pooled"] - trace_scores["mae_pooled"]),
    }
