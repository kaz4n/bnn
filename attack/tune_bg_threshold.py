#!/usr/bin/env python3
"""Sweep S6 background-detection threshold for a model, report pixel-acc + recognition
vs threshold, the optimum, and what choose_threshold() currently picks. Diagnoses why
5x5 background recog lags the paper."""
import argparse, numpy as np
import trace_sim, power_extract, attack_background as bg
from common import load_mnist_test, NCYC, LINE, cyc_to_yx
from evaluate import try_load_golden, recog

ap = argparse.ArgumentParser()
ap.add_argument("--ksize", type=int, default=5)
ap.add_argument("--n", type=int, default=25)
ap.add_argument("--noise", type=float, default=0.1)
args = ap.parse_args()

K = np.load(f"../training/artifacts/model_{args.ksize}x{args.ksize}/layer1_kernels.npy")[:1].astype(np.int8)
xte, yte = load_mnist_test()
golden = try_load_golden()

# per-cycle power (1 kernel) for n eval images (same indices as full run: 300+)
imgs, pws, labels = [], [], []
for n in range(300, 300 + args.n):
    t = trace_sim.simulate_image(xte[n], K, samples_per_cycle=4, noise=args.noise, seed=5000+n)
    pws.append(power_extract.extract_per_cycle(t, 4, NCYC, "sum")[0])
    imgs.append(xte[n]); labels.append(int(yte[n]))

def eval_thr(thr):
    pix, rok = [], 0
    for img, pw, lab in zip(imgs, pws, labels):
        marker = np.zeros((LINE, LINE), np.uint8)
        for c in range(NCYC):
            y, x = cyc_to_yx(c); marker[y, x] = 1 if pw[c] > thr else 0
        pix.append(float((marker == (img > 0).astype(np.uint8)).mean()))
        rok += (recog(golden, marker*255) == lab)
    return np.mean(pix), rok/len(imgs)

def otsu(power, nbins=256):
    c, e = np.histogram(power, bins=nbins)
    c = c.astype(float); p = c / c.sum()
    mids = 0.5 * (e[:-1] + e[1:])
    w0 = np.cumsum(p); w1 = 1 - w0
    m0 = np.cumsum(p * mids) / np.clip(w0, 1e-12, None)
    mt = (p * mids).sum()
    m1 = (mt - np.cumsum(p * mids)) / np.clip(w1, 1e-12, None)
    sigma = w0 * w1 * (m0 - m1) ** 2
    return mids[int(np.nanargmax(sigma))]

allpw = np.concatenate(pws)
print(f"power range: {allpw.min():.2f}..{allpw.max():.2f} mean {allpw.mean():.2f}")
auto = np.mean([bg.choose_threshold(pw) for pw in pws])
otsu_t = np.mean([otsu(pw) for pw in pws])
print(f"choose_threshold() avg pick = {auto:.2f}")
print(f"OTSU avg pick = {otsu_t:.2f}")
print("thr    pixel_acc  recog")
best = (0, 0, 0)
for thr in np.percentile(allpw, np.arange(5, 96, 5)):
    pa, ra = eval_thr(thr)
    print(f"{thr:6.2f}  {pa:.3f}     {ra:.3f}")
    if ra > best[2]: best = (thr, pa, ra)
pa_auto, ra_auto = eval_thr(auto)
pa_ot, ra_ot = eval_thr(otsu_t)
print(f"\nAUTO thr={auto:.2f}: pixel={pa_auto:.3f} recog={ra_auto:.3f}")
print(f"OTSU thr={otsu_t:.2f}: pixel={pa_ot:.3f} recog={ra_ot:.3f}")
print(f"BEST thr={best[0]:.2f}: pixel={best[1]:.3f} recog={best[2]:.3f}")
print(f"(paper recog {args.ksize}x{args.ksize}=" + ("0.646" if args.ksize==5 else "0.816") + ")")
