#!/usr/bin/env python3
"""Active power-template attack for the CW305 MNIST leakage demo.

This script is intentionally split into four stages:

1. build-template:
   Use known profiling images and their real CW305/CW-Lite traces to build a
   power template PT = {(patch pixels, power feature vector)}.

2. make-bundle:
   Create an attack directory containing only traces and capture metadata.  The
   original image and label are written to a separate truth file for scoring.

3. attack:
   Reconstruct images from the trace-only attack bundle using the saved power
   template.  This stage never reads image or label arrays.

4. score:
   Compare recovered images with the separate truth file.

The method follows the active/template idea in Wei et al. (ACSAC 2018): profile
power-to-pixel relations, query candidates from a runtime power vector, and use
overlap consistency to select candidates.  The related-pixel region is adapted to
the current CW305 bitstream: one held 3x3 valid convolution window, not the
paper's line-buffer transition window K*(K+1).
"""
import argparse
import glob
import hashlib
import json
import os
import shutil
import time

import numpy as np
from scipy.spatial import cKDTree

from leakage_first_reconstruct import (
    LINE,
    OUT_SIDE,
    N_WINDOWS,
    all_patch_bits,
    extract_features,
    metrics,
    patch_codes_from_patches,
    patch_matrix_from_image,
)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fp:
        for chunk in iter(lambda: fp.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path):
    with open(path, "r", encoding="utf-8") as fp:
        return json.load(fp)


def save_json(path, payload):
    with open(path, "w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2)


def save_npz_atomic(path, **arrays):
    tmp = path + ".tmp.npz"
    np.savez_compressed(tmp, **arrays)
    if os.path.exists(path):
        os.remove(path)
    os.replace(tmp, path)


def read_capture_params(trace_dir):
    manifest = load_json(os.path.join(trace_dir, "capture_manifest.json"))
    return {
        "manifest": manifest,
        "dwell": int(manifest["dwell"]),
        "spc": int(manifest["samples_per_cycle"]),
        "presamples": int(manifest.get("presamples", 0)),
        "n_kernels": int(manifest["n_kernels"]),
        "feature": "mean_abs",
    }


def file_index(path):
    base = os.path.basename(path)
    return int(base[3:7])


def template_build(args):
    params = read_capture_params(args.traces)
    files = sorted(glob.glob(os.path.join(args.traces, "img*.npz")))
    if len(files) < args.profile_count:
        raise SystemExit(f"need {args.profile_count} profile files, found {len(files)}")
    files = files[:args.profile_count]

    rho_rows = []
    code_rows = []
    source_file_rows = []
    source_window_rows = []
    for path in files:
        d = np.load(path)
        if "image" not in d.files:
            raise SystemExit(f"profile file lacks image ground truth: {path}")
        img = d["image"].astype(np.uint8)
        feat = extract_features(path, params["dwell"], params["spc"],
                                params["presamples"], args.feature)
        patches = patch_matrix_from_image(img)
        codes = patch_codes_from_patches(patches)
        rho_rows.append(feat.T.astype(np.float32))
        code_rows.append(codes.astype(np.uint16))
        source_file_rows.extend([os.path.basename(path)] * N_WINDOWS)
        source_window_rows.extend(range(N_WINDOWS))

    rho = np.vstack(rho_rows).astype(np.float32)
    patch_code = np.concatenate(code_rows).astype(np.uint16)
    patch_bits = all_patch_bits()[patch_code].astype(np.uint8)
    template_manifest = {
        "kind": "cw305_active_power_template",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_trace_dir": os.path.abspath(args.traces),
        "capture_manifest": params["manifest"],
        "profile_count": int(args.profile_count),
        "entries": int(rho.shape[0]),
        "n_kernels": int(rho.shape[1]),
        "related_pixels": "3x3 valid convolution patch",
        "feature": args.feature,
        "normalization": "per-trace robust median/MAD normalization",
        "uses_profile_images": True,
        "attack_stage_uses_ground_truth": False,
        "note": (
            "Profile stage uses known image pixels to build PT. Attack stage should "
            "use only trace-only bundles."
        ),
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    save_npz_atomic(
        args.out,
        rho=rho,
        patch_code=patch_code,
        patch_bits=patch_bits,
        source_file=np.asarray(source_file_rows),
        source_window=np.asarray(source_window_rows, dtype=np.int16),
        manifest_json=np.asarray(json.dumps(template_manifest, indent=2)),
    )
    save_json(args.out + ".json", template_manifest)
    print(json.dumps({
        "template": os.path.abspath(args.out),
        "entries": int(rho.shape[0]),
        "unique_patch_codes": int(np.unique(patch_code).size),
        "profile_count": int(args.profile_count),
    }, indent=2))


def bundle_make(args):
    os.makedirs(args.out, exist_ok=True)
    params = read_capture_params(args.traces)
    files = sorted(glob.glob(os.path.join(args.traces, "img*.npz")))
    files = [f for f in files if file_index(f) >= args.start_index]
    if args.count is not None:
        files = files[:args.count]
    if not files:
        raise SystemExit("no files selected")

    truth = {
        "kind": "cw305_active_template_truth",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_trace_dir": os.path.abspath(args.traces),
        "items": [],
    }
    attack_manifest = {
        "kind": "cw305_trace_only_attack_bundle",
        "created_utc": truth["created_utc"],
        "source_trace_dir": os.path.abspath(args.traces),
        "capture_manifest": params["manifest"],
        "items": [],
        "contains_ground_truth_images": False,
        "contains_labels": False,
    }

    for src in files:
        d = np.load(src)
        keep = {}
        for key in (
            "repeats", "traces", "kernel_ids", "samples_per_cycle", "dwell",
            "samples_per_window", "presamples", "trace_source",
        ):
            if key in d.files:
                keep[key] = d[key]
        dst_name = os.path.basename(src)
        dst = os.path.join(args.out, dst_name)
        save_npz_atomic(dst, **keep)
        attack_manifest["items"].append({
            "file": dst_name,
            "source_file": os.path.basename(src),
            "source_sha256": sha256_file(src),
            "trace_only_sha256": sha256_file(dst),
        })
        truth["items"].append({
            "file": dst_name,
            "source_file": os.path.basename(src),
            "image": d["image"].astype(np.uint8).tolist(),
            "label": int(d["label"]),
        })

    save_json(os.path.join(args.out, "attack_manifest.json"), attack_manifest)
    save_json(args.truth_out, truth)
    print(json.dumps({
        "attack_bundle": os.path.abspath(args.out),
        "truth": os.path.abspath(args.truth_out),
        "items": len(files),
    }, indent=2))


def groups(nk, gsize):
    return [list(range(i, min(i + gsize, nk))) for i in range(0, nk, gsize)]


def build_candidate_search(template_rho, group_defs):
    return [cKDTree(template_rho[:, g]) for g in group_defs]


def candidate_codes_for_query(template_rho, template_codes, query, group_defs, trees,
                              delta, fallback_k, max_candidates):
    group_sets = []
    group_min = []
    radius = float(np.sqrt(delta))
    for g, tree in zip(group_defs, trees):
        idx = np.asarray(tree.query_ball_point(query[g], r=radius), dtype=np.int64)
        if idx.size == 0:
            dist_nn, idx_nn = tree.query(query[g], k=min(fallback_k, template_rho.shape[0]))
            idx = np.atleast_1d(idx_nn).astype(np.int64)
            group_min.append(float(np.min(np.atleast_1d(dist_nn) ** 2)))
        else:
            diff_g = template_rho[np.ix_(idx, g)] - query[g][None, :]
            group_min.append(float(np.min(np.sum(diff_g * diff_g, axis=1))))
        group_sets.append(np.unique(template_codes[idx]))

    cand = group_sets[0]
    for s in group_sets[1:]:
        cand = np.intersect1d(cand, s, assume_unique=True)

    fallback = False
    if cand.size == 0:
        fallback = True
        diff = template_rho - query[None, :]
        dist = np.sum(diff * diff, axis=1)
        idx = np.argsort(dist)[:max(fallback_k * 2, max_candidates)]
        cand = np.unique(template_codes[idx])

    if cand.size > max_candidates:
        # Keep codes whose nearest template entry is closest in full feature space.
        diff = template_rho - query[None, :]
        dist = np.sum(diff * diff, axis=1)
        best = []
        for code in cand:
            mask = template_codes == code
            best.append((float(np.min(dist[mask])), int(code)))
        best.sort()
        cand = np.asarray([code for _, code in best[:max_candidates]], dtype=np.uint16)

    return cand.astype(np.uint16), fallback, group_min


def patch_positions(window):
    y = window // OUT_SIDE
    x = window % OUT_SIDE
    pos = []
    for dy in range(3):
        for dx in range(3):
            pos.append((y + dy) * LINE + (x + dx))
    return np.asarray(pos, dtype=np.int16)


PATCH_POS = [patch_positions(w) for w in range(N_WINDOWS)]
PATCH_BITS = all_patch_bits().astype(np.uint8)


PATCH_NEIGHBORS = []
for _w in range(N_WINDOWS):
    _y, _x = _w // OUT_SIDE, _w % OUT_SIDE
    _nb = []
    for _dy in range(-2, 3):
        for _dx in range(-2, 3):
            _yy, _xx = _y + _dy, _x + _dx
            if 0 <= _yy < OUT_SIDE and 0 <= _xx < OUT_SIDE:
                _nb.append(_yy * OUT_SIDE + _xx)
    PATCH_NEIGHBORS.append(np.asarray(_nb, dtype=np.int32))


def greedy_reconstruct_fast(candidate_lists, max_seed_windows=8):
    """Same greedy overlap-consistency search, without the O(N^2) rescan.

    The selection rule is identical to `greedy_reconstruct`: pick the remaining
    window with the largest overlap with already placed pixels, breaking ties by
    candidate count and then by window index.  Overlap counts are maintained
    incrementally instead of being recomputed for every remaining window.
    """
    counts = np.asarray([len(c) for c in candidate_lists], dtype=np.int32)
    valid = np.flatnonzero(counts > 0)
    if valid.size == 0:
        return np.zeros((LINE, LINE), dtype=np.uint8), np.zeros(N_WINDOWS, dtype=np.uint16)

    cand_arrays = [np.asarray(c, dtype=np.int64) for c in candidate_lists]
    bits_cache = [PATCH_BITS[c].astype(np.float32) if len(c) else None
                  for c in cand_arrays]
    win_index = np.arange(N_WINDOWS, dtype=np.int64)
    seed_order = sorted(valid.tolist(), key=lambda w: (counts[w], w))[:max_seed_windows]
    best_score = None
    best_img = None
    best_codes = None

    for seed_w in seed_order:
        for seed_code in candidate_lists[seed_w]:
            acc = np.zeros(LINE * LINE, dtype=np.float32)
            cnt = np.zeros(LINE * LINE, dtype=np.float32)
            chosen = np.full(N_WINDOWS, 65535, dtype=np.uint16)
            remaining = np.zeros(N_WINDOWS, dtype=bool)
            remaining[valid] = True
            overlap = np.zeros(N_WINDOWS, dtype=np.int32)

            def place(w, code):
                pos = PATCH_POS[w]
                acc[pos] += PATCH_BITS[int(code)]
                cnt[pos] += 1.0
                chosen[w] = int(code)
                remaining[w] = False
                for v in PATCH_NEIGHBORS[w]:
                    if remaining[v]:
                        overlap[v] = int(np.count_nonzero(cnt[PATCH_POS[v]] > 0))

            place(seed_w, int(seed_code))
            while True:
                left = np.flatnonzero(remaining)
                if left.size == 0:
                    break
                # ordering key equals the tuple (-overlap, counts, w)
                key = (-overlap[left].astype(np.int64) * 1000000
                       + counts[left].astype(np.int64) * 1000 + win_index[left])
                w = int(left[int(np.argmin(key))])
                pos = PATCH_POS[w]
                placed = cnt[pos] > 0
                if np.any(placed):
                    est = acc[pos[placed]] / np.maximum(cnt[pos[placed]], 1.0)
                    diff = bits_cache[w][:, placed] - est[None, :]
                    code = int(cand_arrays[w][int(np.argmin(np.sum(diff * diff, axis=1)))])
                else:
                    code = int(cand_arrays[w][0])
                place(w, code)

            img_soft = np.divide(acc, cnt, out=np.zeros_like(acc), where=cnt > 0)
            score = 0.0
            for w in valid:
                bits = PATCH_BITS[int(chosen[w])].astype(np.float32)
                est = img_soft[PATCH_POS[w]]
                score += float(np.sum((bits - est) ** 2))
            if best_score is None or score < best_score:
                best_score = score
                best_img = (img_soft.reshape(LINE, LINE) >= 0.5).astype(np.uint8)
                best_codes = chosen.copy()

    return best_img, best_codes


def greedy_reconstruct(candidate_lists, max_seed_windows=8):
    counts = np.asarray([len(c) for c in candidate_lists], dtype=np.int32)
    valid = np.flatnonzero(counts > 0)
    if valid.size == 0:
        return np.zeros((LINE, LINE), dtype=np.uint8), np.zeros(N_WINDOWS, dtype=np.uint16)

    seed_order = sorted(valid.tolist(), key=lambda w: (counts[w], w))[:max_seed_windows]
    best_score = None
    best_img = None
    best_codes = None

    for seed_w in seed_order:
        for seed_code in candidate_lists[seed_w]:
            acc = np.zeros(LINE * LINE, dtype=np.float32)
            cnt = np.zeros(LINE * LINE, dtype=np.float32)
            chosen = np.full(N_WINDOWS, 65535, dtype=np.uint16)
            remaining = set(valid.tolist())

            def place(w, code):
                bits = PATCH_BITS[int(code)].astype(np.float32)
                pos = PATCH_POS[w]
                acc[pos] += bits
                cnt[pos] += 1.0
                chosen[w] = int(code)
                remaining.discard(w)

            place(seed_w, int(seed_code))
            while remaining:
                # Select the next window with largest overlap with already placed pixels.
                scored = []
                for w in remaining:
                    overlap = int(np.count_nonzero(cnt[PATCH_POS[w]] > 0))
                    scored.append((-overlap, counts[w], w))
                scored.sort()
                w = scored[0][2]
                pos = PATCH_POS[w]
                placed = cnt[pos] > 0
                if np.any(placed):
                    est = acc[pos[placed]] / np.maximum(cnt[pos[placed]], 1.0)
                    costs = []
                    for code in candidate_lists[w]:
                        bits = PATCH_BITS[int(code)].astype(np.float32)
                        costs.append(float(np.sum((bits[placed] - est) ** 2)))
                    code = int(candidate_lists[w][int(np.argmin(costs))])
                else:
                    code = int(candidate_lists[w][0])
                place(w, code)

            img_soft = np.divide(acc, cnt, out=np.zeros_like(acc), where=cnt > 0)
            score = 0.0
            for w in valid:
                bits = PATCH_BITS[int(chosen[w])].astype(np.float32)
                est = img_soft[PATCH_POS[w]]
                score += float(np.sum((bits - est) ** 2))
            if best_score is None or score < best_score:
                best_score = score
                best_img = (img_soft.reshape(LINE, LINE) >= 0.5).astype(np.uint8)
                best_codes = chosen.copy()

    return best_img, best_codes


def attack_run(args):
    templ = np.load(args.template, allow_pickle=False)
    template_rho = templ["rho"].astype(np.float32)
    template_codes = templ["patch_code"].astype(np.uint16)
    template_manifest = json.loads(str(templ["manifest_json"]))
    attack_manifest = load_json(os.path.join(args.bundle, "attack_manifest.json"))
    if attack_manifest.get("contains_ground_truth_images"):
        raise SystemExit("attack bundle says it contains ground truth; refusing")

    cap = attack_manifest["capture_manifest"]
    dwell = int(cap["dwell"])
    spc = int(cap["samples_per_cycle"])
    presamples = int(cap.get("presamples", 0))
    group_defs = groups(template_rho.shape[1], args.group_size)
    trees = build_candidate_search(template_rho, group_defs)
    os.makedirs(args.out, exist_ok=True)
    rows = []

    for item in attack_manifest["items"]:
        path = os.path.join(args.bundle, item["file"])
        d = np.load(path)
        forbidden = {"image", "label", "original", "truth", "true_codes"}
        present_forbidden = sorted(forbidden.intersection(d.files))
        if present_forbidden:
            raise SystemExit(f"attack file contains forbidden keys {present_forbidden}: {path}")

        feat = extract_features(path, dwell, spc, presamples, args.feature).T
        candidate_lists = []
        fallback_count = 0
        cand_sizes = []
        group_min_all = []
        for w in range(N_WINDOWS):
            cand, fallback, group_min = candidate_codes_for_query(
                template_rho, template_codes, feat[w], group_defs, trees,
                args.delta, args.fallback_k, args.max_candidates)
            candidate_lists.append(cand)
            fallback_count += int(fallback)
            cand_sizes.append(int(cand.size))
            group_min_all.extend(group_min)

        reconstruct = (greedy_reconstruct if getattr(args, 'slow_reconstruct', False)
                       else greedy_reconstruct_fast)
        recon, pred_codes = reconstruct(candidate_lists, args.max_seed_windows)
        out_name = f"recovered_{item['file'][3:7]}.npz"
        save_npz_atomic(
            os.path.join(args.out, out_name),
            recovered=recon.astype(np.uint8),
            pred_codes=pred_codes.astype(np.uint16),
            candidate_sizes=np.asarray(cand_sizes, dtype=np.int16),
            source_attack_file=np.asarray(item["file"]),
        )
        rows.append({
            "file": item["file"],
            "recovered_file": out_name,
            "mean_candidates": float(np.mean(cand_sizes)),
            "median_candidates": float(np.median(cand_sizes)),
            "max_candidates": int(np.max(cand_sizes)),
            "fallback_windows": int(fallback_count),
            "mean_group_min_dist": float(np.mean(group_min_all)),
        })
        print(f"{item['file']} mean_cands={rows[-1]['mean_candidates']:.2f} "
              f"fallback={fallback_count}")

    payload = {
        "kind": "cw305_active_power_template_attack_result",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "template": os.path.abspath(args.template),
        "template_manifest": template_manifest,
        "attack_bundle": os.path.abspath(args.bundle),
        "attack_manifest": attack_manifest,
        "parameters": {
            "delta": args.delta,
            "group_size": args.group_size,
            "fallback_k": args.fallback_k,
            "max_candidates": args.max_candidates,
            "max_seed_windows": args.max_seed_windows,
            "feature": args.feature,
        },
        "attack_stage_inputs": "trace-only npz files plus template; no image/label arrays",
        "images": rows,
        "summary": {
            "n_images": len(rows),
            "mean_candidates": float(np.mean([r["mean_candidates"] for r in rows])),
            "mean_fallback_windows": float(np.mean([r["fallback_windows"] for r in rows])),
        },
    }
    save_json(os.path.join(args.out, "attack_summary.json"), payload)


def score_run(args):
    truth = load_json(args.truth)
    truth_by_file = {item["file"]: item for item in truth["items"]}
    attack_summary = load_json(os.path.join(args.results, "attack_summary.json"))
    rows = []
    for item in attack_summary["images"]:
        rec = np.load(os.path.join(args.results, item["recovered_file"]))["recovered"]
        t = truth_by_file[item["file"]]
        img = np.asarray(t["image"], dtype=np.uint8)
        true_codes = patch_codes_from_patches(patch_matrix_from_image(img))
        pred_codes = np.load(os.path.join(args.results, item["recovered_file"]))["pred_codes"]
        row = metrics(rec, img)
        row["patch_exact"] = float(np.mean(pred_codes == true_codes))
        row["label"] = int(t["label"])
        row["file"] = item["file"]
        row.update({
            "mean_candidates": item["mean_candidates"],
            "fallback_windows": item["fallback_windows"],
        })
        rows.append(row)

    keys = [k for k, v in rows[0].items() if isinstance(v, (int, float, np.floating))]
    summary = {k: float(np.mean([r[k] for r in rows])) for k in keys
               if k not in ("label",)}
    payload = {
        "kind": "cw305_active_power_template_score",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "results": os.path.abspath(args.results),
        "truth": os.path.abspath(args.truth),
        "images": rows,
        "summary": summary,
    }
    save_json(os.path.join(args.results, "score_summary.json"), payload)
    print(json.dumps(summary, indent=2))


def score_recognition_run(args):
    import sys

    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "training")))
    from train_golden_mlp import build_mlp

    truth = load_json(args.truth)
    truth_by_file = {item["file"]: item for item in truth["items"]}
    attack_summary = load_json(os.path.join(args.results, "attack_summary.json"))

    model = build_mlp()
    model.load_weights(os.path.abspath(os.path.join(
        os.path.dirname(__file__), "..", "training", "artifacts",
        "golden_mlp", "weights.weights.h5")))

    def recog(img01):
        img01 = np.asarray(img01, dtype=np.float32)
        # Trace attack reconstructs binary images.  Map 0/1 back to MNIST-like
        # 0/255 before the same [-1,1] normalization used during training.
        x = ((img01 * 255.0) / 127.5 - 1.0)[None, ..., None]
        return int(model.predict(x, verbose=0).argmax())

    rows = []
    orig_ok = 0
    rec_ok = 0
    for item in attack_summary["images"]:
        truth_item = truth_by_file[item["file"]]
        rec_name = item["recovered_file"]
        rec = np.load(os.path.join(args.results, rec_name))["recovered"]
        orig = np.asarray(truth_item["image"], dtype=np.uint8)
        label = int(truth_item["label"])
        po = recog(orig)
        pr = recog(rec)
        orig_ok += int(po == label)
        rec_ok += int(pr == label)
        rows.append({
            "file": rec_name,
            "attack_file": item["file"],
            "label": label,
            "orig_pred": po,
            "recovered_pred": pr,
            "orig_ok": po == label,
            "recovered_ok": pr == label,
        })

    payload = {
        "kind": "cw305_active_power_template_recognition_score",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "n": len(rows),
        "orig_recognition_acc": orig_ok / len(rows),
        "recovered_recognition_acc": rec_ok / len(rows),
        "rows": rows,
    }
    save_json(os.path.join(args.results, "recognition_score.json"), payload)
    print(json.dumps({
        "n": payload["n"],
        "orig_recognition_acc": payload["orig_recognition_acc"],
        "recovered_recognition_acc": payload["recovered_recognition_acc"],
    }, indent=2))


def export_images(args):
    import matplotlib.pyplot as plt

    truth = load_json(args.truth)
    truth_by_file = {item["file"]: item for item in truth["items"]}
    score = load_json(os.path.join(args.results, "score_summary.json"))
    rows = score["images"][:args.count]
    if not rows:
        raise SystemExit("no scored rows")
    fig, axes = plt.subplots(len(rows), 2, figsize=(4.2, 1.85 * len(rows)))
    if len(rows) == 1:
        axes = np.asarray([axes])
    for r, axrow in zip(rows, axes):
        truth_item = truth_by_file[r["file"]]
        orig = np.asarray(truth_item["image"], dtype=np.uint8)
        rec_name = "recovered_" + r["file"][3:7] + ".npz"
        rec = np.load(os.path.join(args.results, rec_name))["recovered"]
        axrow[0].imshow(orig, cmap="gray_r", vmin=0, vmax=1)
        axrow[0].set_title(f"Truth {r['label']}")
        axrow[1].imshow(rec, cmap="gray_r", vmin=0, vmax=1)
        axrow[1].set_title(f"Recovered F1={r['foreground_f1']:.2f}")
        for ax in axrow:
            ax.axis("off")
    fig.suptitle("Trace-only active-template reconstruction", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fig.savefig(args.out, dpi=180)
    print(os.path.abspath(args.out))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("build-template")
    p.add_argument("--traces", required=True)
    p.add_argument("--profile-count", type=int, default=30)
    p.add_argument("--feature", choices=["mean_abs", "rms", "p2p", "cycle_abs"], default="mean_abs")
    p.add_argument("--out", required=True)
    p.set_defaults(func=template_build)

    p = sub.add_parser("make-bundle")
    p.add_argument("--traces", required=True)
    p.add_argument("--start-index", type=int, default=30)
    p.add_argument("--count", type=int, default=None)
    p.add_argument("--out", required=True)
    p.add_argument("--truth-out", required=True)
    p.set_defaults(func=bundle_make)

    p = sub.add_parser("attack")
    p.add_argument("--template", required=True)
    p.add_argument("--bundle", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--feature", choices=["mean_abs", "rms", "p2p", "cycle_abs"], default="mean_abs")
    p.add_argument("--delta", type=float, default=1.0)
    p.add_argument("--group-size", type=int, default=3)
    p.add_argument("--fallback-k", type=int, default=128)
    p.add_argument("--max-candidates", type=int, default=48)
    p.add_argument("--max-seed-windows", type=int, default=8)
    p.add_argument("--slow-reconstruct", action="store_true",
                   help="use the original O(N^2) reference implementation")
    p.set_defaults(func=attack_run)

    p = sub.add_parser("score")
    p.add_argument("--results", required=True)
    p.add_argument("--truth", required=True)
    p.set_defaults(func=score_run)

    p = sub.add_parser("score-recognition")
    p.add_argument("--results", required=True)
    p.add_argument("--truth", required=True)
    p.set_defaults(func=score_recognition_run)

    p = sub.add_parser("export-figure")
    p.add_argument("--results", required=True)
    p.add_argument("--truth", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--count", type=int, default=6)
    p.set_defaults(func=export_images)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
