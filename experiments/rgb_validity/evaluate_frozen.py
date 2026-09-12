#!/usr/bin/env python3
"""Task-decomposed calibrated validation of FROZEN reconstructions.

Reads existing prediction arrays. Trains nothing, touches no hardware, loads no traces,
and changes not one predicted pixel. The point predictions and their MAE are exactly what
they were; what changes is the scientific output -- from a pooled point score to a set of
per-task statements, each of which can be withheld independently.

PROTOCOL, fixed before looking at any assessment number:
  populations   summed mode; natural and tinted-control kept separate (never pooled)
  tasks         PRIMARY full-resolution Y, Cb, Cr; SECONDARY 8x8 coarse Y
  alpha         0.10 per task, separately -- three 90% statements are NOT a joint 90%
  split         scene-disjoint, 50/50 calibration/assessment, fixed seed, saved membership
  baseline      the exported analytic prior, calibrated by the identical protocol
  usefulness    a radius counts only if coverage held AND it is tighter than the prior's

RETROSPECTIVE, AND LABELLED SO. These rows are the same held-out examples that informed
earlier corrections and paper claims in this project. Re-splitting old test data does not
make it fresh: the frozen predictor and the score definitions were not chosen
independently of these cases. Coverage computed here is therefore exploratory evidence
about the reporting formulation, NOT a prospective guarantee. Establishing the latter
requires examples with a clean usage history.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from . import calibration as C
from . import source_audit as SA
from . import tasks as T
from .schema import FrozenPredictionBatch

R = "experiments/rgb_sca/results"

CELL_SOURCES = {
    "summed": (f"{R}/hardware_20260912_rev3", "host/traces_rgb_full"),
    "serial": (f"{R}/hardware_20260912_rev3", "host/traces_rgb_full"),
    "parallel": (f"{R}/hardware_parallel_rev3", "host/traces_rgb_parallel"),
}


def load_batch(mode, group):
    """Load one frozen export, with source identity verified by two routes."""
    base, trace_dir = CELL_SOURCES[mode]
    npz = f"{base}/images/{mode}/reconstructions_{group}.npz"
    if not os.path.exists(npz):
        return None, {"routes_agree": False, "verdict": f"missing export {npz}"}
    audit = SA.audit_cell(trace_dir, mode, group, npz)
    if not audit["routes_agree"]:
        return None, audit
    z = np.load(npz)
    prior = z["prior_only"] if "prior_only" in z.files else None
    if prior is not None and prior.size == 0:
        prior = None
    batch = FrozenPredictionBatch(
        mode=mode, group=group,
        truth=z["truth"].astype(np.float64),
        prediction=z["trace"].astype(np.float64),
        prior_prediction=None if prior is None else prior.astype(np.float64),
        source_id=np.asarray(audit["source_id"]),
        scene_id=np.asarray(audit["scene_id"]),
        session_id=None,
        provenance={"npz": npz, "trace_dir": trace_dir, "audit": audit})
    return batch, audit


def split_by_scene(scene_id, frac_cal=0.5, seed=0):
    """Partition example rows by SCENE, never by row. Returns (cal_idx, assess_idx)."""
    scenes = np.unique(scene_id)
    order = np.random.default_rng(seed).permutation(len(scenes))
    n_cal = int(round(frac_cal * len(scenes)))
    cal_scenes = set(scenes[order[:n_cal]].tolist())
    cal = np.array([i for i, s in enumerate(scene_id) if s in cal_scenes], dtype=int)
    ass = np.array([i for i, s in enumerate(scene_id) if s not in cal_scenes], dtype=int)
    SA.assert_scene_disjoint(scene_id[cal], scene_id[ass])
    return cal, ass


def seed_stability(batch, alpha, frac_cal, seeds):
    """How much of the verdict table is the split seed rather than the data?

    POST-HOC, and a diagnostic only: seed 0 remains the declared primary result. The
    motivation is visible in that result -- natural/Cr has ratio 0.982, so it is tighter
    than the prior and fails only because its coverage landed at 0.810. Realized coverage
    of a conformal radius varies across splits by a few points on its own, so a cell
    decided by that margin is decided by the seed. This measures how often each verdict
    survives a re-split, so a table of six verdicts is not read as six findings when some
    of them are coin flips.

    The frozen predictions are untouched; only the calibration/assessment partition moves.
    """
    rows = {}
    for name in list(T.PRIMARY) + list(T.SECONDARY):
        e = T.task_scores(batch.prediction, batch.truth, name)
        ep = (None if batch.prior_prediction is None
              else T.task_scores(batch.prior_prediction, batch.truth, name))
        useful, covs, ratios = [], [], []
        for s in seeds:
            cal, ass = split_by_scene(batch.scene_id, frac_cal, s)
            q = C.conformal_radius(e[cal], alpha)
            cov = C.evaluate_coverage(q["radius"], e[ass], alpha)
            covs.append(cov["coverage"])
            if ep is None:
                continue
            qp = C.conformal_radius(ep[cal], alpha)
            ratios.append(q["radius"] / qp["radius"] if qp["radius"] else np.nan)
            useful.append(bool(q["radius"] < qp["radius"]
                               and not cov["materially_undercovers"]))
        rows[name] = {
            "n_seeds": len(seeds),
            "fraction_of_seeds_useful": (float(np.mean(useful)) if useful else None),
            "coverage_min_max": [float(np.min(covs)), float(np.max(covs))],
            "ratio_min_max": ([float(np.nanmin(ratios)), float(np.nanmax(ratios))]
                              if ratios else None),
            "stable": (None if not useful else bool(np.mean(useful) in (0.0, 1.0))),
            # Guard against the obvious misreading. A high fraction says the verdict does
            # not depend on which scenes were drawn; it says nothing about the SIZE of the
            # advantage. natural/Cb fires in most splits because its ratio sits just under
            # 1.0 in all of them -- consistently, but by ~2%, which the ratio bootstrap
            # cannot resolve from zero. Consistency and magnitude are separate questions,
            # and only a ratio range entirely below 1.0 speaks to the second.
            "note": ("fraction_of_seeds_useful measures seed-sensitivity of the verdict, "
                     "NOT strength of evidence; check ratio_min_max lies wholly below "
                     "1.0 before reading a high fraction as support"),
            "ratio_range_excludes_one": (
                None if not ratios else bool(np.nanmax(ratios) < 1.0)),
        }
    return rows


def evaluate_cell(batch, audit, alpha, frac_cal, seed):
    cal, ass = split_by_scene(batch.scene_id, frac_cal, seed)
    cell = {
        "n": int(len(batch.truth)), "n_cal": int(len(cal)), "n_assess": int(len(ass)),
        "split": {"kind": "scene-disjoint", "seed": seed, "frac_cal": frac_cal,
                  "calibration_scene_ids": batch.scene_id[cal].tolist(),
                  "assessment_scene_ids": batch.scene_id[ass].tolist()},
        "source_audit": {k: audit[k] for k in
                         ("routes_agree", "hash_lookup_misses",
                          "hash_collisions_in_capture_set", "verdict")},
        "artifact_hashes": batch.hashes(),
        "n_prediction_values_outside_unit_interval": batch.n_clipped,
        "has_prior_baseline": batch.prior_prediction is not None,
        "tasks": {},
    }

    print(f"  {'task':10s} {'radius':>8s} {'prior':>8s} {'ratio':>6s} "
          f"{'ratio 95%':>13s} {'cover':>6s} {'95% Wilson':>15s} "
          f"{'covers?':>8s} {'useful?':>8s}")
    for name in list(T.PRIMARY) + list(T.SECONDARY):
        e = T.task_scores(batch.prediction, batch.truth, name)
        q = C.conformal_radius(e[cal], alpha)
        cov = C.evaluate_coverage(q["radius"], e[ass], alpha)
        entry = {"primary": name in T.PRIMARY, "calibration": q, "coverage": cov,
                 "assess_mean_error_255": float(e[ass].mean())}

        if batch.prior_prediction is not None:
            ep = T.task_scores(batch.prior_prediction, batch.truth, name)
            qp = C.conformal_radius(ep[cal], alpha)
            covp = C.evaluate_coverage(qp["radius"], ep[ass], alpha)
            entry["prior"] = {"calibration": qp, "coverage": covp,
                              "assess_mean_error_255": float(ep[ass].mean())}
            tighter = bool(q["finite"] and qp["finite"] and q["radius"] < qp["radius"])
            entry["tighter_than_prior"] = tighter
            entry["radius_ratio_vs_prior"] = (
                float(q["radius"] / qp["radius"]) if qp["radius"] else None)
            # A radius means nothing on its own: 255 covers every image. Useful requires
            # BOTH that coverage held and that the set beats the no-input baseline.
            entry["useful"] = bool(tighter and not cov["materially_undercovers"])
            ru = radius_ratio_uncertainty(e, ep, cal, alpha, seed=seed)
            entry["post_hoc_ratio_uncertainty"] = ru
            rci = (f"[{ru['ratio_ci_95'][0]:.2f},{ru['ratio_ci_95'][1]:.2f}]"
                   if ru else "--")
            print(f"  {name:10s} {q['radius']:8.2f} {qp['radius']:8.2f} "
                  f"{entry['radius_ratio_vs_prior']:6.3f} {rci:>13s} "
                  f"{cov['coverage']:6.3f} "
                  f"[{cov['wilson_95'][0]:.3f},{cov['wilson_95'][1]:.3f}] "
                  f"{str(not cov['materially_undercovers']):>8s} "
                  f"{str(entry['useful']):>8s}")
        else:
            # No baseline export -> usefulness is undecidable. Not assumed either way.
            entry["useful"] = None
            entry["note"] = ("no prior baseline in this export; a radius without a "
                             "no-input comparison cannot be called useful")
            print(f"  {name:10s} {q['radius']:8.2f} {'--':>8s} {'--':>6s} "
                  f"{'--':>13s} {cov['coverage']:6.3f} "
                  f"[{cov['wilson_95'][0]:.3f},{cov['wilson_95'][1]:.3f}] "
                  f"{str(not cov['materially_undercovers']):>8s} {'undecid':>8s}")
        cell["tasks"][name] = entry
    return cell


def radius_ratio_uncertainty(e, ep, cal, alpha, n_boot=2000, seed=0):
    """Resampling spread of the radius ratio. ADDED AFTER seeing the pilot output.

    The declared rule compares two point radii. That was under-specified: on 100
    calibration examples a ratio of 0.98 and a ratio of 1.02 are the same measurement,
    but the rule calls one useful and the other not. This does NOT change the declared
    verdict -- it reports whether the margin behind that verdict is resolvable at all.

    Resampling is over calibration examples, which are scenes here, so the unit is the
    independent one. The resampled statistic is a high order statistic (k=91 of 100),
    where the bootstrap is approximate: read the interval as indicative of whether a
    margin is resolvable, not as an exact confidence statement.
    """
    rng = np.random.default_rng(seed)
    ratios = []
    for _ in range(n_boot):
        b = cal[rng.integers(0, len(cal), len(cal))]
        q = C.conformal_radius(e[b], alpha)["radius"]
        qp = C.conformal_radius(ep[b], alpha)["radius"]
        if np.isfinite(q) and np.isfinite(qp) and qp > 0:
            ratios.append(q / qp)
    if not ratios:
        return None
    lo, hi = np.percentile(ratios, [2.5, 97.5])
    return {"ratio_ci_95": [float(lo), float(hi)], "n_boot": len(ratios),
            "resolvably_tighter": bool(hi < 1.0),
            "note": ("resolvably_tighter=False means the ratio interval spans 1.0: the "
                     "declared point comparison decided this cell on a margin the data "
                     "cannot resolve")}


def decide(cell):
    """Mechanical application of the predeclared categories. No post-hoc judgement."""
    est, wit, fragile = [], [], []
    for name, e in cell["tasks"].items():
        if not e["primary"]:
            continue
        (est if e.get("useful") else wit).append(name)
        ru = e.get("post_hoc_ratio_uncertainty")
        if e.get("useful") and ru and not ru["resolvably_tighter"]:
            fragile.append(name)
    return {
        "established_primary_tasks": est,
        "withheld_primary_tasks": wit,
        # Post-hoc, and kept separate from the declared verdict rather than folded into
        # it: the rule was declared on point radii, so its output stands as computed.
        "established_on_a_margin_the_data_cannot_resolve": fragile,
        "reading": ("Established = the calibrated set covered on scene-disjoint "
                    "assessment examples AND was tighter than the no-input prior's, in "
                    "the retrospective sense declared at the top of this file. Withheld "
                    "tasks are NOT shown to be impossible -- they are unsupported by "
                    "these frozen outputs under this protocol."),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--modes", nargs="*", default=["summed"],
                    help="minimum viable experiment is summed only")
    ap.add_argument("--groups", nargs="*", default=["control", "natural"],
                    help="populations are kept separate, never pooled")
    ap.add_argument("--alpha", type=float, default=0.10, help="per task, not joint")
    ap.add_argument("--frac-cal", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0, help="the declared primary split")
    ap.add_argument("--stability-seeds", type=int, default=20,
                    help="post-hoc re-splits; 0 or 1 disables the diagnostic")
    ap.add_argument("--out", default="experiments/rgb_validity/results/frozen_pilot")
    args = ap.parse_args()

    out = {
        "kind": "rgb_validity_frozen_pilot",
        "status": ("RETROSPECTIVE AND EXPLORATORY -- not a prospective coverage "
                   "guarantee; these assessment rows were previously inspected"),
        "hardware_used": False, "retrained": False, "predictions_modified": False,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "protocol": {
            "alpha_per_task": args.alpha, "joint_guarantee_across_tasks": False,
            "split": "scene-disjoint", "frac_cal": args.frac_cal, "seed": args.seed,
            "populations_pooled": False,
            "tasks_primary": list(T.PRIMARY), "tasks_secondary": list(T.SECONDARY),
        },
        "cells": {}, "blocked": [], "verdicts": {},
    }

    for mode in args.modes:
        for group in args.groups:
            key = f"{mode}/{group}"
            batch, audit = load_batch(mode, group)
            if batch is None:
                out["blocked"].append({"cell": key, "reason": audit["verdict"]})
                print(f"\nBLOCKED {key}: {audit['verdict']}")
                continue
            print(f"\n=== {key}   n={len(batch.truth)}  alpha={args.alpha} per task ===")
            print(f"  identity: {audit['verdict']}  "
                  f"(hash misses {audit['hash_lookup_misses']}, "
                  f"collisions {audit['hash_collisions_in_capture_set']})")
            cell = evaluate_cell(batch, audit, args.alpha, args.frac_cal, args.seed)
            print(f"  split: {cell['n_cal']} calibration / {cell['n_assess']} "
                  f"assessment scenes, disjoint")
            if args.stability_seeds > 1:
                stab = seed_stability(batch, args.alpha, args.frac_cal,
                                      range(args.stability_seeds))
                cell["post_hoc_seed_stability"] = stab
                print(f"  post-hoc stability over {args.stability_seeds} re-splits "
                      f"(seed 0 remains the declared result):")
                for name, r in stab.items():
                    f = r["fraction_of_seeds_useful"]
                    print(f"    {name:10s} useful in {f:5.0%} of splits   "
                          f"coverage {r['coverage_min_max'][0]:.2f}-"
                          f"{r['coverage_min_max'][1]:.2f}   "
                          f"ratio {r['ratio_min_max'][0]:.2f}-"
                          f"{r['ratio_min_max'][1]:.2f}"
                          f"{'  <- ratio range below 1' if r['ratio_range_excludes_one'] else ''}")
            out["cells"][key] = cell
            out["verdicts"][key] = decide(cell)

    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, "results.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=1, default=float)

    print("\n=== decision ===")
    for k, v in out["verdicts"].items():
        print(f"  {k:18s} established: {v['established_primary_tasks'] or 'none'}"
              f"   withheld: {v['withheld_primary_tasks'] or 'none'}")
        if v["established_on_a_margin_the_data_cannot_resolve"]:
            print(f"  {'':18s} ^ post-hoc: "
                  f"{v['established_on_a_margin_the_data_cannot_resolve']} passed only "
                  f"on a radius ratio whose 95% interval spans 1.0")
    print(f"\n{out['status']}")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
