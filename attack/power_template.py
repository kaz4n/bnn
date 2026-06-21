"""
P3 -- S7 Image Reconstruction via Power Template (active adversary).

Profiling: for each profiling image, for each cycle c, store
    (related-pixel values Px_c  [K*(K+1)],  power-feature-vector rho_c [n_kernels]).
Attack: for an unknown image's per-cycle rho', find template entries whose rho is within
delta (searched per kernel-group, then intersected) -> pixel-value candidates per cycle.
Reconstruct: Algorithm 2 (greedy, consistency/variance-minimizing) then average.

Defaults follow the paper S7.3: 9 kernels, 3 groups of 3, delta=1.0.
"""
import numpy as np
from common import LINE, NCYC, related_pixels


def build_template(images, per_cycle_power_fn, kernels):
    """
    images: list of 28x28 uint8. per_cycle_power_fn(img)-> [n_kernels, 784] per-cycle pw.
    Returns dict with rho [M, nk], pxvals [M, P], cyc [M] (cycle index per entry).
    """
    K = kernels.shape[1]
    P = K * (K + 1)
    rho_all, px_all, cyc_all = [], [], []
    for img in images:
        pc = per_cycle_power_fn(img)                     # [nk, 784]
        for c in range(NCYC):
            _, vals = related_pixels(img, c, K)
            px_all.append(vals); rho_all.append(pc[:, c]); cyc_all.append(c)
    return {"rho": np.asarray(rho_all, np.float32),
            "pxvals": np.asarray(px_all, np.int32),
            "cyc": np.asarray(cyc_all, np.int32), "K": K, "P": P}


def _groups(nk, gsize):
    return [list(range(i, i + gsize)) for i in range(0, (nk // gsize) * gsize, gsize)]


def generate_candidates(template, rho_attack, c, delta=1.0, gsize=3):
    """
    rho_attack: [nk] measured power-feature-vector at cycle c of the attack image.
    Returns array of candidate Px-value vectors [n_cand, P] (intersection over groups).
    """
    rho = template["rho"]; px = template["pxvals"]
    nk = rho.shape[1]
    groups = _groups(nk, gsize)
    sets = []
    for Km in groups:
        d = ((rho[:, Km] - rho_attack[Km]) ** 2).sum(axis=1)   # group distance (Eq.)
        idx = np.where(d < delta)[0]
        sets.append({tuple(v) for v in px[idx]})
    if not sets:
        return np.empty((0, template["P"]), np.int32)
    inter = set.intersection(*sets) if len(sets) > 1 else sets[0]
    if not inter:                                               # fall back to union
        inter = set.union(*sets)
    return np.array(sorted(inter), dtype=np.int32) if inter else \
        np.empty((0, template["P"]), np.int32)


def reconstruct(cand_per_cycle, K, seeds=5, rng=None):
    """
    Algorithm 2 (faithful adaptation). cand_per_cycle: list length 784; each is
    [n_cand, P] candidate Px-value vectors for that cycle. Returns 28x28 float image.

    Per cycle c, the P candidate values map to fixed positions = related_pixels(.,c).
    We pick one candidate per cycle to minimize cross-cycle pixel-value variance, then
    average the chosen candidates per pixel (paper: average of selected candidates).
    """
    rng = rng or np.random.default_rng(0)
    P = None
    # precompute positions per cycle (independent of image content)
    dummy = np.zeros((LINE, LINE), np.int32)
    positions = []
    for c in range(NCYC):
        pos, _ = related_pixels(dummy, c, K)
        positions.append(pos)
        if P is None and len(cand_per_cycle[c]):
            P = cand_per_cycle[c].shape[1]

    best_img, best_var = None, np.inf
    order = [c for c in range(NCYC) if len(cand_per_cycle[c])]
    for _ in range(seeds):
        acc = np.zeros((LINE, LINE), np.float64)
        cnt = np.zeros((LINE, LINE), np.float64)
        rng.shuffle(order)
        for c in order:
            cands = cand_per_cycle[c]
            pos = positions[c]
            # discrepancy of each candidate vs current estimate over in-bounds positions
            best_k, best_d = 0, np.inf
            for ci, cv in enumerate(cands):
                d = 0.0; n = 0
                for (yy, xx), val in zip(pos, cv):
                    if 0 <= yy < LINE and 0 <= xx < LINE and cnt[yy, xx] > 0:
                        d += (val - acc[yy, xx] / cnt[yy, xx]) ** 2; n += 1
                d = d / n if n else 0.0
                if d < best_d:
                    best_d, best_k = d, ci
            cv = cands[best_k]
            for (yy, xx), val in zip(pos, cv):
                if 0 <= yy < LINE and 0 <= xx < LINE:
                    acc[yy, xx] += val; cnt[yy, xx] += 1
        img = np.divide(acc, cnt, out=np.zeros_like(acc), where=cnt > 0)
        # variance proxy = mean residual already minimized; recompute total
        var = float(np.nanmean((acc / np.where(cnt > 0, cnt, 1) - img) ** 2))
        if var < best_var:
            best_var, best_img = var, img
    return np.clip(best_img, 0, 255) if best_img is not None else np.zeros((LINE, LINE))


def pixel_distance(recovered, original):
    """Paper Eq.7: mean L2 per pixel between recovered and original (0..255 scale)."""
    return float(np.sqrt(((recovered.astype(float) - original.astype(float)) ** 2)).mean())
