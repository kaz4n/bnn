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
from scipy.spatial import cKDTree
from common import LINE, NCYC, related_pixels


def build_template2(images, pcs, K):
    """images: list 28x28 uint8; pcs: parallel list of [n_kernels,784] per-cycle power
    (from sim OR real CW captures). The template is source-agnostic."""
    P = K * (K + 1)
    rho_all, px_all, cyc_all = [], [], []
    for img, pc in zip(images, pcs):
        for c in range(NCYC):
            _, vals = related_pixels(img, c, K)
            px_all.append(vals); rho_all.append(pc[:, c]); cyc_all.append(c)
    return {"rho": np.asarray(rho_all, np.float32),
            "pxvals": np.asarray(px_all, np.int32),
            "cyc": np.asarray(cyc_all, np.int32), "K": K, "P": P}


def build_template(images, per_cycle_power_fn, kernels):
    """Convenience wrapper: compute per-cycle power per image then build_template2."""
    pcs = [per_cycle_power_fn(img) for img in images]
    return build_template2(images, pcs, kernels.shape[1])


def _groups(nk, gsize):
    return [list(range(i, i + gsize)) for i in range(0, (nk // gsize) * gsize, gsize)]


def build_search(template, gsize=3):
    """Build one KDTree per kernel-group over the template's rho columns (fast ball
    search) plus integer codes for each distinct related-pixel vector, so group
    intersection is pure-numpy (np.intersect1d) instead of Python set-of-tuples."""
    rho = template["rho"]; nk = rho.shape[1]
    groups = _groups(nk, gsize)
    trees = [cKDTree(rho[:, Km]) for Km in groups]
    # factorize px rows -> integer code per template entry + representative row per code
    pv = np.ascontiguousarray(template["pxvals"])
    void = pv.view(np.dtype((np.void, pv.dtype.itemsize * pv.shape[1]))).ravel()
    _, first, inv = np.unique(void, return_index=True, return_inverse=True)
    return {"trees": trees, "groups": groups,
            "codes": inv.astype(np.int64),          # (N,) code per entry
            "rep_row": template["pxvals"][first],    # (n_uniq, P) row per code
            "P": template["P"]}


def generate_candidates(template, search, rho_attack, c, delta=1.0,
                        metric="paper_l1", empty_policy="strict"):
    """
    rho_attack: [nk] measured power-feature-vector at cycle c of the attack image.
    Returns array of candidate Px-value vectors [n_cand, P] (intersection over groups).
    metric="paper_l1" follows the paper's written distance. Each rho_i is a scalar, so
    sum_i ||rho_i-rho'_i||_2 is L1 across the kernel group and the search radius is delta.
    metric="squared_l2" preserves the earlier adapted implementation.

    empty_policy="strict" follows the paper: an empty intersection is a no-match.
    empty_policy="union" preserves the earlier adapted fallback.
    """
    if metric == "paper_l1":
        p, r = 1, float(delta)
    elif metric == "squared_l2":
        p, r = 2, float(np.sqrt(delta))
    else:
        raise ValueError(f"unknown candidate metric: {metric}")
    if empty_policy not in ("strict", "union"):
        raise ValueError(f"unknown empty-intersection policy: {empty_policy}")

    codes = search["codes"]
    code_sets = []
    for tree, Km in zip(search["trees"], search["groups"]):
        idx = tree.query_ball_point(rho_attack[Km], r, p=p)    # fast radius search
        code_sets.append(np.unique(codes[np.asarray(idx, dtype=np.int64)])
                         if idx else np.empty(0, np.int64))
    if not code_sets:
        return np.empty((0, search["P"]), np.int32)
    inter = code_sets[0]
    for s in code_sets[1:]:
        inter = np.intersect1d(inter, s, assume_unique=True)
    if inter.size == 0 and empty_policy == "union":             # adapted fallback
        inter = np.unique(np.concatenate(code_sets)) if code_sets else inter
    return search["rep_row"][inter].astype(np.int32)


def reconstruct(cand_per_cycle, K, seeds=6, rng=None):
    """
    Algorithm 2 (faithful). cand_per_cycle: list length 784; each [n_cand, P] candidate
    Px-value vectors for that cycle. Returns 28x28 float image.

    Paper Algorithm 2 grows the image by OVERLAP (SelectCycle = max overlap with the
    placed region) and at each step picks the candidate of minimal discrepancy with the
    current estimate, then averages the selected candidates per pixel. Scan order already
    maximizes overlap (consecutive line-buffer windows overlap), so we grow in scan order
    from varied seed starts and keep the selector with minimal total consistency error.
    """
    rng = rng or np.random.default_rng(0)
    # precompute, per cycle, the flat pixel indices in-bounds + their slot in the cand vec
    posflat, slot = [], []
    dummy = np.zeros((LINE, LINE), np.int32)
    for c in range(NCYC):
        pos, _ = related_pixels(dummy, c, K)
        pf, sl = [], []
        for k, (yy, xx) in enumerate(pos):
            if 0 <= yy < LINE and 0 <= xx < LINE:
                pf.append(yy * LINE + xx); sl.append(k)
        posflat.append(np.asarray(pf, np.int64)); slot.append(np.asarray(sl, np.int64))

    valid = [c for c in range(NCYC) if len(cand_per_cycle[c])]
    if not valid:
        return np.zeros((LINE, LINE))
    starts = [valid[int(i * len(valid) / seeds)] for i in range(seeds)]

    best_img, best_score = None, np.inf
    for c0 in starts:
        acc = np.zeros(NCYC, np.float64); cnt = np.zeros(NCYC, np.float64)
        i0 = valid.index(c0)
        order = valid[i0:] + valid[:i0]                  # scan order rotated to seed
        chosen = {}
        for c in order:
            cands = cand_per_cycle[c]; pf = posflat[c]; sl = slot[c]
            cvals = cands[:, sl]                           # (n_cand, n_inb)
            placed = cnt[pf] > 0
            if not placed.any() or len(cands) == 1:
                k = 0 if len(cands) == 1 else int(rng.integers(len(cands)))
            else:
                est = acc[pf][placed] / cnt[pf][placed]
                d = ((cvals[:, placed] - est) ** 2).sum(axis=1)   # min discrepancy
                k = int(d.argmin())
            np.add.at(acc, pf, cvals[k]); np.add.at(cnt, pf, 1.0)
            chosen[c] = k
        img = np.divide(acc, cnt, out=np.zeros(NCYC), where=cnt > 0)
        # consistency score: total squared error of selected candidates vs final image
        score = 0.0
        for c in valid:
            pf = posflat[c]; sl = slot[c]
            score += float(((cand_per_cycle[c][chosen[c]][sl] - img[pf]) ** 2).sum())
        if score < best_score:
            best_score, best_img = score, img.reshape(LINE, LINE)
    return np.clip(best_img, 0, 255)


def pixel_distance(recovered, original):
    """Paper Eq.7: mean L2 per pixel between recovered and original (0..255 scale)."""
    return float(np.sqrt(((recovered.astype(float) - original.astype(float)) ** 2)).mean())
