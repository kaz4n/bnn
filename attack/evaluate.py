#!/usr/bin/env python3
"""
P3 -- end-to-end attack evaluation (runs NOW on simulated traces; swap to real CW traces
by pointing load at host .npz instead of the simulator).

Pipeline per the paper:
  S5-replacement front-end (power_extract) -> S6 background detection -> S7 power template
  -> metrics: pixel accuracy/distance + recognition accuracy via golden MLP.

Usage (demo):
  python evaluate.py --ksize 3 --n-profile 20 --n-eval 8
Full faithful run (slow; matches paper sizes):
  python evaluate.py --ksize 3 --n-profile 300 --n-eval 200 --n-kernels 9
"""
import argparse, json, os, sys
import numpy as np

import trace_sim, power_extract
import attack_background as bg
import power_template as pt
from common import load_mnist_test, NCYC

SPC = 4          # samples/cycle (sim); on the bench = CW adc_mul


def make_power_fn(kernels, noise, seed0):
    """Returns f(img, seed)-> [nk,784] per-cycle power via the real front-end."""
    def f(img, seed):
        traces = trace_sim.simulate_image(img, kernels, samples_per_cycle=SPC,
                                          noise=noise, seed=seed)
        return power_extract.extract_per_cycle(traces, SPC, NCYC, reduce="sum")
    return f


def load_kernels(ksize, path, n_kernels):
    if path and os.path.exists(path):
        k = np.load(path)[:n_kernels].astype(np.int8); src = path
    else:
        rng = np.random.default_rng(0)
        k = rng.choice([-1, 1], size=(n_kernels, ksize, ksize)).astype(np.int8)
        src = "random (P2 weights not found)"
    return k, src


def try_load_golden():
    try:
        sys.path.insert(0, os.path.join("..", "training"))
        from train_golden_mlp import build_mlp
        m = build_mlp()
        m.load_weights(os.path.join("..", "training", "artifacts",
                                    "golden_mlp", "weights.weights.h5"))
        return m
    except Exception as e:
        print(f"[recognition] golden MLP unavailable ({e}); reporting pixel metrics only")
        return None


def recog(model, img28):
    if model is None:
        return None
    x = (img28.astype("float32") / 127.5 - 1.0)[None, ..., None]
    return int(model.predict(x, verbose=0).argmax())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ksize", type=int, choices=[3, 5], default=3)
    ap.add_argument("--n-profile", type=int, default=20)
    ap.add_argument("--n-eval", type=int, default=8)
    ap.add_argument("--n-kernels", type=int, default=9)
    ap.add_argument("--delta", type=float, default=0.1,
                    help="template match radius (delta^0.5) on z-scored rho; paper delta=1.0 "
                         "in raw units maps to ~0.1 after per-kernel normalization")
    ap.add_argument("--gsize", type=int, default=3)
    ap.add_argument("--metric", default="squared_l2", choices=["squared_l2", "paper_l1"],
                    help="candidate distance metric; squared_l2+delta0.1+union is the "
                         "validated config that reproduces paper-level S7 (paper_l1+strict "
                         "at delta0.1 starves candidates -> ~0.24)")
    ap.add_argument("--empty-policy", default="union", choices=["union", "strict"],
                    help="empty group-intersection handling; union falls back to the group "
                         "union (validated), strict returns no-match (paper-literal)")
    ap.add_argument("--noise", type=float, default=2.0,
                    help="per-sample Gaussian trace noise (sim stand-in for CW-Lite SNR)")
    ap.add_argument("--no-normalize", action="store_true",
                    help="disable per-kernel z-score of power features before S7")
    ap.add_argument("--kernels", default=None,
                    help="P2 layer1_kernels.npy; omit -> random kernels")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    if args.kernels is None:
        cand = f"../training/artifacts/model_{args.ksize}x{args.ksize}/layer1_kernels.npy"
        args.kernels = cand
    kernels, ksrc = load_kernels(args.ksize, args.kernels, args.n_kernels)
    print(f"[setup] K={args.ksize} kernels={kernels.shape} src={ksrc}")

    xte, yte = load_mnist_test()
    prof = list(range(args.n_profile))
    ev = list(range(args.n_profile, args.n_profile + args.n_eval))

    pfn = make_power_fn(kernels, args.noise, 0)
    golden = try_load_golden()

    # ---- S7 build template (profiling) ----
    print(f"[template] building from {len(prof)} images ...")
    prof_pcs = [pfn(xte[i], seed=1000 + i) for i in prof]     # deterministic per-image seed
    # Normalize per-kernel power features (z-score) using the profiling set so the template
    # match radius delta is invariant to the ABSOLUTE leakage scale (FANOUT gain, bench LNA
    # gain, ADC range). Mirrors run_on_hardware.py --normalize-rho; lets the paper's fixed
    # delta port across hardware without re-tuning.
    rho_mu = rho_sd = None
    if not args.no_normalize:
        stack = np.concatenate(prof_pcs, axis=1)
        rho_mu = stack.mean(axis=1, keepdims=True)
        rho_sd = stack.std(axis=1, keepdims=True)
        rho_sd = np.where(rho_sd < 1e-9, 1.0, rho_sd)
        prof_pcs = [(p - rho_mu) / rho_sd for p in prof_pcs]

    def nrm(pc):
        return pc if rho_mu is None else (pc - rho_mu) / rho_sd

    tmpl = pt.build_template2([xte[i] for i in prof], prof_pcs, args.ksize)
    print(f"[template] {tmpl['rho'].shape[0]} entries")
    search = pt.build_search(tmpl, args.gsize)         # KDTree per kernel-group

    # RESUMABLE: per-image metrics appended to metrics.jsonl; already-done images are
    # skipped on relaunch. Robust to the machine killing the long CPU job mid-run.
    mpath = os.path.join(args.out, "metrics.jsonl")
    done = {}
    if os.path.exists(mpath):
        with open(mpath) as f:
            for line in f:
                line = line.strip()
                if line:
                    r = json.loads(line); done[r["idx"]] = r
        print(f"[resume] {len(done)} images already done, skipping them")

    for n in ev:
        if n in done:
            continue
        img = xte[n]; label = int(yte[n])
        pc = pfn(img, seed=5000 + n)                     # attack-time traces
        pc_s7 = nrm(pc)

        bimg, marker, thr = bg.recover_background(pc[0])           # S6 (raw power)
        bg_pixel = bg.pixel_accuracy(marker, img)

        cands = [pt.generate_candidates(tmpl, search, pc_s7[:, c], c, args.delta,
                                        metric=args.metric, empty_policy=args.empty_policy)
                 for c in range(NCYC)]                              # S7 (normalized rho)
        rimg = pt.reconstruct(cands, args.ksize)
        tm_d = pt.pixel_distance(rimg, img)

        ro = recog(golden, img); rb = recog(golden, bimg); rt = recog(golden, rimg)

        np.savez(os.path.join(args.out, f"recovered_{n:04d}.npz"),
                 original=img, background=bimg, template=rimg, label=label)
        rec = {"idx": n, "label": label, "bg_pixel_acc": bg_pixel,
               "tm_pixel_dist": tm_d, "recog_orig": ro, "recog_bg": rb, "recog_tm": rt}
        with open(mpath, "a") as f:                                 # append = durable
            f.write(json.dumps(rec) + "\n")
        done[n] = rec
        print(f"  img {n} (label {label}): bg_pix={bg_pixel:.3f} "
              f"tm_dist={tm_d:.2f} recog(orig/bg/tm)={ro}/{rb}/{rt}", flush=True)

    recs = [done[n] for n in ev if n in done]
    ne = len(recs)
    has_g = golden is not None
    res = {"ksize": args.ksize, "n_profile": args.n_profile, "n_eval": ne,
           "kernel_src": ksrc, "images": recs, "summary": {
        "bg_pixel_acc_mean": float(np.mean([r["bg_pixel_acc"] for r in recs])),
        "tm_pixel_dist_mean": float(np.mean([r["tm_pixel_dist"] for r in recs])),
        "recog_acc_orig": sum(r["recog_orig"] == r["label"] for r in recs)/ne if has_g else None,
        "recog_acc_background": sum(r["recog_bg"] == r["label"] for r in recs)/ne if has_g else None,
        "recog_acc_template": sum(r["recog_tm"] == r["label"] for r in recs)/ne if has_g else None,
        "paper_targets": {"bg_pixel_acc": 0.862, "recog_background": 0.816,
                          "recog_template": 0.898} if args.ksize == 3 else
                         {"bg_pixel_acc": 0.746, "recog_background": 0.646,
                          "recog_template": 0.790}}}
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(res, f, indent=2)
    print("\n[summary]", json.dumps(res["summary"], indent=2))


if __name__ == "__main__":
    main()
