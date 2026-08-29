#!/usr/bin/env python3
"""
TVLA (Test Vector Leakage Assessment) -- Welch's t-test, fixed-vs-random, on RAW samples.

Answers the ONE question the corr>=0.1 metric could not: is there ANY image-dependent
leakage in this measurement channel? It does not assume a leakage model (magnitude,
Hamming, per-cycle) -- it just asks whether the trace distribution differs between a FIXED
input and RANDOM inputs. |t| > 4.5 at any sample == leakage present (standard threshold).

Run this on the bench BEFORE spending days on bitstream rebuilds:
  - |t| >> 4.5 somewhere  -> leakage EXISTS; the attack's failure is exploitation, not a
    dead channel. Pursue fabric-quiescing + deterministic capture + Lever A.
  - |t| stays < 4.5       -> the conv datapath is buried below the channel floor on this
    bench. Power-only will not get there; EM probe or a quieter target is required.

Two modes:
  --sim               self-test on the trace simulator (no hardware). Shows TVLA fires on
                      clean sim and collapses under global-activity dilution+jitter --
                      i.e. it reproduces, and would have diagnosed, the hardware failure.
  --fixed DIR --random DIR   real captures. Each DIR holds .npz with a 'traces' array
                      [n_kernels, n_samples]; ideally SINGLE (un-averaged) traces, many of
                      them. Use --kernel to pick which kernel's trace to assess.
"""
import argparse, glob, os
import numpy as np


def welch_t(fixed, random):
    """Per-sample Welch t between two [n_traces, n_samples] groups."""
    mf, mr = fixed.mean(0), random.mean(0)
    vf, vr = fixed.var(0, ddof=1), random.var(0, ddof=1)
    nf, nr = fixed.shape[0], random.shape[0]
    denom = np.sqrt(vf / nf + vr / nr)
    denom = np.where(denom < 1e-30, 1e-30, denom)
    return (mf - mr) / denom


def report(t, thr=4.5):
    amax = float(np.max(np.abs(t)))
    n_leak = int(np.sum(np.abs(t) > thr))
    verdict = "LEAKAGE PRESENT" if amax > thr else "no leakage above threshold"
    print(f"  max|t| = {amax:8.2f}   samples over {thr}: {n_leak}/{t.size}   -> {verdict}")
    return amax, n_leak


def load_group(d, kernel):
    traces = []
    for f in sorted(glob.glob(os.path.join(d, "*.npz"))):
        arr = np.load(f)["traces"]
        traces.append(np.asarray(arr[kernel], dtype=np.float64))
    if not traces:
        raise SystemExit(f"no .npz in {d}")
    m = min(len(t) for t in traces)
    return np.stack([t[:m] for t in traces], 0)


def sim_selftest():
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import trace_sim
    from common import load_mnist_test
    try:
        XT, _ = load_mnist_test()
    except Exception:
        XT = np.load(os.path.join("..", "host", "mnist_test.npz"))["images"]
    kernels = np.load(os.path.join("..", "training", "artifacts", "model_3x3",
                                   "layer1_kernels.npy"))[:9].astype(np.int8)
    N = 60
    fixed_img = XT[0]
    rand_imgs = [XT[100 + i] for i in range(N)]

    def group(imgs, **kw):
        rows = []
        for i, im in enumerate(imgs):
            tr = trace_sim.simulate_image(im, kernels, samples_per_cycle=4, noise=2.0,
                                          seed=10_000 + i, **kw)
            rows.append(tr[0])              # kernel 0 raw trace
        return np.stack(rows, 0)

    print("TVLA self-test (kernel 0, fixed image vs 60 random images):")
    print("[clean sim] strong leakage ->")
    F = group([fixed_img] * N); R = group(rand_imgs)
    report(welch_t(F, R))
    print("[dilution=100x, jitter=5% -- the hardware regime] leakage DEGRADES but persists ->")
    F = group([fixed_img] * N, dilution=100, dilution_jitter=0.05)
    R = group(rand_imgs, dilution=100, dilution_jitter=0.05)
    report(welch_t(F, R))
    print("\nReading -- TVLA is NECESSARY, NOT SUFFICIENT:\n"
          "  * The data-independent global term cancels in the fixed-vs-random MEAN\n"
          "    difference, so TVLA still fires (t drops 92->~26) even in the regime where\n"
          "    the correlation/template ATTACK fails (corr ~0.09). TVLA is robust to exactly\n"
          "    the common-mode contaminant that swamps recovery.\n"
          "  * So on the bench: |t| < 4.5 everywhere == DEAD channel, stop (power-only\n"
          "    cannot win; need EM or a quieter target). |t| > 4.5 == channel ALIVE but you\n"
          "    still face the dilution/exploitation gap -- pursue fabric-quiescing +\n"
          "    deterministic capture + Lever A, and re-measure recovery, not just t.\n"
          "  See memory dilution-rootcause.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", action="store_true", help="self-test on the simulator (no HW)")
    ap.add_argument("--fixed", help="dir of fixed-input capture .npz")
    ap.add_argument("--random", help="dir of random-input capture .npz")
    ap.add_argument("--kernel", type=int, default=0)
    ap.add_argument("--thr", type=float, default=4.5)
    args = ap.parse_args()
    if args.sim:
        sim_selftest(); return
    if not (args.fixed and args.random):
        raise SystemExit("need --sim, or both --fixed and --random")
    F = load_group(args.fixed, args.kernel)
    R = load_group(args.random, args.kernel)
    print(f"fixed {F.shape}  random {R.shape}  kernel {args.kernel}")
    report(welch_t(F, R), args.thr)


if __name__ == "__main__":
    main()
