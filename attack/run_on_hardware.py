#!/usr/bin/env python3
"""
P3 on REAL hardware traces -- run S6 background detection + S7 power template on the
.npz files captured by ../host/cw305_bnn_capture.py (CW305 + ChipWhisperer-Lite).

Identical attack code as evaluate.py; only the trace SOURCE differs (disk, not sim).
This is the script you run once you have the bench + captures.

Usage:
  python run_on_hardware.py --ksize 3 --traces-dir ../host/traces \
      --n-profile 300 --n-eval 200 --out results_hw_3x3
"""
import argparse, glob, hashlib, json, os, time
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np

import power_extract
import attack_background as bg
import power_template as pt
from common import NCYC
from evaluate import try_load_golden, recog          # reuse golden-MLP recognition


def _extract_one(task):
    f, tag, cache_dir, cfg_hash, extract_cfg = task
    cache_path = None
    if cache_dir:
        stem = os.path.splitext(os.path.basename(f))[0]
        cache_path = os.path.join(cache_dir, f"{stem}_{cfg_hash}.npz")
        if os.path.exists(cache_path):
            d = np.load(cache_path)
            return tag, f, d["pc"], d["image"], int(d["label"]), 0.0, True
    t0 = time.perf_counter()
    pc, img, label, _ = power_extract.load_and_extract(
        f, reduce=extract_cfg["reduce"], frontend=extract_cfg["frontend"],
        sample_rate_hz=extract_cfg["sample_rate"],
        lowpass_cutoff_hz=extract_cfg["lowpass_cutoff"],
        dc_cutoff_hz=extract_cfg["dc_cutoff"],
        do_dc_restore=extract_cfg["do_dc_restore"],
        do_lowpass=extract_cfg["do_lowpass"],
        do_align=extract_cfg["do_align"],
        do_curve_fit=extract_cfg["do_curve_fit"],
        tail_cycles=extract_cfg["tail_cycles"],
        min_curve_fit_spc=extract_cfg["min_curve_fit_spc"])
    dt = time.perf_counter() - t0
    if cache_path:
        tmp_path = cache_path + ".tmp.npz"
        np.savez(tmp_path, pc=pc, image=img, label=label)
        os.replace(tmp_path, cache_path)
    return tag, f, pc, img, label, dt, False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ksize", type=int, choices=[3, 5], required=True)
    ap.add_argument("--traces-dir", required=True, help="dir of img####.npz captures")
    ap.add_argument("--n-profile", type=int, default=300)
    ap.add_argument("--n-eval", type=int, default=200)
    ap.add_argument("--delta", type=float, default=0.1,
                    help="template match radius on z-scored rho (paper delta=1.0 in raw "
                         "units ~= 0.1 after per-kernel normalization)")
    ap.add_argument("--gsize", type=int, default=3)
    ap.add_argument("--metric", default="squared_l2", choices=["squared_l2", "paper_l1"],
                    help="candidate distance metric; squared_l2+delta0.1+union is the "
                         "validated config that reproduces paper-level S7 in sim")
    ap.add_argument("--empty-policy", default="union", choices=["union", "strict"],
                    help="empty group-intersection handling (union=validated fallback)")
    ap.add_argument("--invert", action="store_true",
                    help="negate per-cycle power before the attack. Required for a DROOP/RO "
                         "sensor (more switching -> LOWER count): flips S6 so high=foreground. "
                         "S7 is unaffected (z-score is sign-symmetric when template+attack "
                         "are both negated).")
    ap.add_argument("--reduce", default="sum",
                    choices=["sum", "mean", "peak", "ptp", "rms", "abs_sum",
                             "template_amp", "residual_rms"])
    ap.add_argument("--frontend", default="auto", choices=["auto", "direct", "paper"],
                    help="power extraction frontend; auto uses paper S5 only for multi-sample non-RO traces")
    ap.add_argument("--sample-rate", type=float, default=None,
                    help="raw trace sample rate for paper S5 extraction; default reads capture metadata")
    ap.add_argument("--lowpass-cutoff", type=float, default=None,
                    help="paper S5 low-pass cutoff; default 60 MHz, clamped below Nyquist if needed")
    ap.add_argument("--dc-cutoff", type=float, default=250.0,
                    help="paper S5 equivalent measurement high-pass cutoff for DC restore")
    ap.add_argument("--no-dc-restore", action="store_true")
    ap.add_argument("--no-lowpass", action="store_true")
    ap.add_argument("--no-align", action="store_true")
    ap.add_argument("--no-curve-fit", action="store_true")
    ap.add_argument("--tail-cycles", type=int, default=3,
                    help="number of cycles of fitted trailing power to subtract")
    ap.add_argument("--min-curve-fit-spc", type=int, default=8,
                    help="minimum samples/cycle needed before nonlinear RC curve fitting runs")
    ap.add_argument("--extract-cache", default=None,
                    help="cache extracted per-cycle power here; default is <out>/extract_cache")
    ap.add_argument("--no-extract-cache", action="store_true",
                    help="disable extraction cache")
    ap.add_argument("--jobs", type=int, default=1,
                    help="parallel extraction jobs; useful for curve-fit processing")
    ap.add_argument("--no-normalize-rho", action="store_true",
                    help="disable z-score of per-kernel power features (normalization is ON by "
                         "default so the fixed delta is invariant to the bench's absolute power scale)")
    ap.add_argument("--out", default="results_hw")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    cache_dir = None if args.no_extract_cache else (args.extract_cache or os.path.join(args.out, "extract_cache"))
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)

    files = sorted(glob.glob(os.path.join(args.traces_dir, "*.npz")))
    need = args.n_profile + args.n_eval
    assert len(files) >= need, f"need {need} captures, found {len(files)}"
    prof_f, ev_f = files[:args.n_profile], files[args.n_profile:need]

    extract_cfg = {
        "reduce": args.reduce,
        "frontend": args.frontend,
        "sample_rate": args.sample_rate,
        "lowpass_cutoff": args.lowpass_cutoff,
        "dc_cutoff": args.dc_cutoff,
        "do_dc_restore": not args.no_dc_restore,
        "do_lowpass": not args.no_lowpass,
        "do_align": not args.no_align,
        "do_curve_fit": not args.no_curve_fit,
        "tail_cycles": args.tail_cycles,
        "min_curve_fit_spc": args.min_curve_fit_spc,
    }
    cfg_hash = hashlib.sha256(json.dumps(extract_cfg, sort_keys=True).encode()).hexdigest()[:12]

    def load_many(tagged_files):
        tasks = [(f, tag, cache_dir, cfg_hash, extract_cfg) for tag, f in tagged_files]
        out = [None] * len(tasks)
        if max(1, args.jobs) == 1:
            for i, task in enumerate(tasks):
                tag, f, pc, img, label, dt, cached = _extract_one(task)
                suffix = "cache" if cached else f"{dt:.1f}s"
                print(f"[extract] {tag} {os.path.basename(f)} {suffix}", flush=True)
                out[i] = (pc, img, label)
            return out
        with ProcessPoolExecutor(max_workers=args.jobs) as ex:
            futs = {ex.submit(_extract_one, task): i for i, task in enumerate(tasks)}
            for fut in as_completed(futs):
                i = futs[fut]
                tag, f, pc, img, label, dt, cached = fut.result()
                suffix = "cache" if cached else f"{dt:.1f}s"
                print(f"[extract] {tag} {os.path.basename(f)} {suffix}", flush=True)
                out[i] = (pc, img, label)
        return out

    normalize = not args.no_normalize_rho          # z-score rho by default
    print(f"[hw] profiling from {len(prof_f)} captures ...")
    prof = load_many([(f"profile {i + 1}/{len(prof_f)}", f) for i, f in enumerate(prof_f)])
    if args.invert:
        prof = [(-p[0], p[1], p[2]) for p in prof]
    if normalize:
        stack = np.concatenate([p[0] for p in prof], axis=1)
        rho_mu = stack.mean(axis=1, keepdims=True)
        rho_sd = stack.std(axis=1, keepdims=True)
        rho_sd = np.where(rho_sd < 1e-9, 1.0, rho_sd)
        prof = [((p[0] - rho_mu) / rho_sd, p[1], p[2]) for p in prof]
        print("[hw] normalized rho features with profiling mean/std", flush=True)
    else:
        rho_mu = rho_sd = None
    tmpl = pt.build_template2([p[1] for p in prof], [p[0] for p in prof], args.ksize)
    search = pt.build_search(tmpl, args.gsize)
    print(f"[hw] template {tmpl['rho'].shape[0]} entries")

    golden = try_load_golden()
    res = {"ksize": args.ksize, "source": args.traces_dir,
           "frontend": args.frontend, "delta": args.delta,
           "normalize_rho": bool(normalize),
           "extract_cfg": extract_cfg, "images": []}
    bg_pix, tm_dist, bg_ok, tm_ok, or_ok = [], [], 0, 0, 0

    ev = load_many([(f"eval {i + 1}/{len(ev_f)}", f) for i, f in enumerate(ev_f)])
    if args.invert:
        ev = [(-e[0], e[1], e[2]) for e in ev]
    for i, (f, loaded) in enumerate(zip(ev_f, ev)):
        pc, img, label = loaded
        pc_s7 = pc
        if normalize:
            pc_s7 = (pc - rho_mu) / rho_sd
        bimg, marker, thr = bg.recover_background(pc[0])
        bg_pix.append(bg.pixel_accuracy(marker, img))
        cands = [pt.generate_candidates(tmpl, search, pc_s7[:, c], c, args.delta,
                                        metric=args.metric, empty_policy=args.empty_policy)
                 for c in range(NCYC)]
        rimg = pt.reconstruct(cands, args.ksize)
        tm_dist.append(pt.pixel_distance(rimg, img))
        ro, rb, rt = recog(golden, img), recog(golden, bimg), recog(golden, rimg)
        or_ok += (ro == label); bg_ok += (rb == label); tm_ok += (rt == label)
        np.savez(os.path.join(args.out, os.path.basename(f)),
                 original=img, background=bimg, template=rimg, label=label)
        res["images"].append({"file": os.path.basename(f), "label": label,
                              "bg_pixel_acc": bg_pix[-1], "tm_pixel_dist": tm_dist[-1],
                              "recog_orig": ro, "recog_bg": rb, "recog_tm": rt})
        print(f"  {os.path.basename(f)} (label {label}): bg_pix={bg_pix[-1]:.3f} "
              f"tm_dist={tm_dist[-1]:.2f} recog(o/b/t)={ro}/{rb}/{rt}")

    ne = len(ev_f)
    res["summary"] = {
        "bg_pixel_acc_mean": float(np.mean(bg_pix)),
        "tm_pixel_dist_mean": float(np.mean(tm_dist)),
        "recog_acc_orig": or_ok / ne if golden else None,
        "recog_acc_background": bg_ok / ne if golden else None,
        "recog_acc_template": tm_ok / ne if golden else None,
        "paper_targets": {"recog_background": 0.816, "recog_template": 0.898}
                         if args.ksize == 3 else
                         {"recog_background": 0.646, "recog_template": 0.790},
    }
    with open(os.path.join(args.out, "summary.json"), "w") as fp:
        json.dump(res, fp, indent=2)
    print("\n[hw summary]", json.dumps(res["summary"], indent=2))


if __name__ == "__main__":
    main()
