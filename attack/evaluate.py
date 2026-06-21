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
    ap.add_argument("--delta", type=float, default=1.0)
    ap.add_argument("--gsize", type=int, default=3)
    ap.add_argument("--noise", type=float, default=0.5)
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
    tmpl = pt.build_template([xte[i] for i in prof],
                             lambda im: pfn(im, seed=1000 + hash(im.tobytes()) % 9999),
                             kernels)
    print(f"[template] {tmpl['rho'].shape[0]} entries")

    res = {"ksize": args.ksize, "n_profile": args.n_profile, "n_eval": args.n_eval,
           "kernel_src": ksrc, "images": []}
    bg_pix, tm_dist, bg_recog_ok, tm_recog_ok, orig_recog_ok = [], [], 0, 0, 0

    for n in ev:
        img = xte[n]; label = int(yte[n])
        pc = pfn(img, seed=5000 + n)                     # attack-time traces

        # S6 background detection (one kernel)
        bimg, marker, thr = bg.recover_background(pc[0])
        bg_pix.append(bg.pixel_accuracy(marker, img))

        # S7 power template
        cands = [pt.generate_candidates(tmpl, pc[:, c], c, args.delta, args.gsize)
                 for c in range(NCYC)]
        rimg = pt.reconstruct(cands, args.ksize)
        tm_dist.append(pt.pixel_distance(rimg, img))

        # recognition
        ro = recog(golden, img); rb = recog(golden, bimg); rt = recog(golden, rimg)
        orig_recog_ok += (ro == label)
        bg_recog_ok += (rb == label); tm_recog_ok += (rt == label)

        np.savez(os.path.join(args.out, f"recovered_{n:04d}.npz"),
                 original=img, background=bimg, template=rimg, label=label)
        res["images"].append({"idx": n, "label": label,
                              "bg_pixel_acc": bg_pix[-1], "tm_pixel_dist": tm_dist[-1],
                              "recog_orig": ro, "recog_bg": rb, "recog_tm": rt})
        print(f"  img {n} (label {label}): bg_pix={bg_pix[-1]:.3f} "
              f"tm_dist={tm_dist[-1]:.2f} recog(orig/bg/tm)={ro}/{rb}/{rt}")

    ne = len(ev)
    res["summary"] = {
        "bg_pixel_acc_mean": float(np.mean(bg_pix)),
        "tm_pixel_dist_mean": float(np.mean(tm_dist)),
        "recog_acc_orig": orig_recog_ok / ne if golden else None,
        "recog_acc_background": bg_recog_ok / ne if golden else None,
        "recog_acc_template": tm_recog_ok / ne if golden else None,
        "paper_targets": {"bg_pixel_acc": 0.862, "recog_background": 0.816,
                          "recog_template": 0.898} if args.ksize == 3 else
                         {"bg_pixel_acc": 0.746, "recog_background": 0.646,
                          "recog_template": 0.790},
    }
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(res, f, indent=2)
    print("\n[summary]", json.dumps(res["summary"], indent=2))


if __name__ == "__main__":
    main()
