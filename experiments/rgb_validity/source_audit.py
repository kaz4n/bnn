"""Recover and VERIFY source identity for frozen prediction exports.

The methodology review named this the evidentiary bottleneck: the exports save
group-local row positions (`test_indices`), not original source IDs, and equal positions
across two modes do not prove equal images. Everything else is blocked until identity is
established, so identity is established twice, independently, and the two routes are
required to agree.

  route A -- index arithmetic. Replays the selection analyze_hardware.py performed:
             load the capture, map shard indices through the image-set group array, take
             the group subset, then index it by the exported positions. This trusts that
             the loader is deterministic and that nothing upstream reordered.

  route B -- content hash. Hashes each exported reference image and looks it up in the
             capture set. This trusts nothing about bookkeeping; it asks the pixels.

Route B alone would be enough if hashes were unique, so route A is not redundant
decoration: agreement between a bookkeeping-derived answer and a content-derived answer
is what rules out a silent ordering bug. Disagreement is a hard failure, not a warning.
"""
from __future__ import annotations

import hashlib
import os

import numpy as np


def _hash_index(images: np.ndarray) -> dict:
    """sha256 of each uint8 image -> row. Collisions are reported, never silently kept."""
    out = {}
    for i in range(len(images)):
        h = hashlib.sha256(np.ascontiguousarray(images[i]).tobytes()).hexdigest()
        out.setdefault(h, []).append(i)
    dupes = {h: v for h, v in out.items() if len(v) > 1}
    return {h: v[0] for h, v in out.items()}, dupes


def audit_cell(trace_dir: str, mode: str, group: str, npz_path: str,
               image_set: str = "host/rgb_capture_images.npz") -> dict:
    """Recover source IDs for one frozen export and cross-check the two routes."""
    from experiments.rgb_sca import rgb_generator as G

    d = np.load(image_set)
    allimg, allgrp = d["images"], d["group"]
    h2row, dupes = _hash_index(allimg)

    z = np.load(npz_path)
    truth, te = z["truth"], z["test_indices"]

    # route A -- replay the selection
    _f, _i, idx, _m = G.load_rgb_shards(trace_dir, mode)
    sel = np.where(allgrp[idx] == group)[0]
    route_a = idx[sel][te]

    # route B -- ask the pixels
    route_b, misses = [], 0
    for i in range(len(truth)):
        key = hashlib.sha256(
            np.round(np.clip(truth[i], 0, 1) * 255).astype(np.uint8).tobytes()).hexdigest()
        r = h2row.get(key, -1)
        route_b.append(r)
        misses += (r == -1)
    route_b = np.asarray(route_b)

    agree = bool(np.array_equal(route_a, route_b))
    src = route_a if agree else None
    return {
        "mode": mode, "group": group, "npz": npz_path,
        "n_rows": int(len(truth)),
        "hash_collisions_in_capture_set": len(dupes),
        "hash_lookup_misses": int(misses),
        "routes_agree": agree,
        "source_id": None if src is None else src.tolist(),
        # Each capture row is a distinct source image (verified separately), so scene ==
        # source here. Kept as its own field because that equality is a property of THIS
        # dataset, not a general truth -- rolled or re-photographed variants would break it.
        "scene_id": None if src is None else src.tolist(),
        "session_id": None,
        "session_note": ("single capture session; no session variable exists in these "
                         "artifacts, so cross-session transport cannot be assessed"),
        "verdict": ("identity established by two independent routes" if agree
                    else "BLOCKED: index arithmetic and content hash disagree"),
    }


def assert_scene_disjoint(a_ids, b_ids, label_a="calibration", label_b="assessment"):
    """Hard failure if two roles share a scene. A shared scene invalidates coverage."""
    overlap = set(np.asarray(a_ids).tolist()) & set(np.asarray(b_ids).tolist())
    if overlap:
        raise ValueError(
            f"{label_a} and {label_b} share {len(overlap)} scene(s), e.g. "
            f"{sorted(overlap)[:5]}. Split membership must be disjoint at the scene "
            "level or the coverage statement is void.")
    return True
