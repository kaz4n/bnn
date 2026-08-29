#!/usr/bin/env python3
"""
Bench check: does the FANOUT leakage amplifier actually amplify MEASURED power?

The FINAL_RESULTS "16x amplifier had no effect -> fundamental wall" conclusion is only
valid if FANOUT=16 truly produced ~16x more data-dependent switching in silicon. HLS
unrolled 16 lanes (RTL confirms), but downstream Vivado synthesis can still merge identical
lanes, and the shared line-buffer/mux may dominate power. This script measures it.

Metric = DATA-DEPENDENT signal energy in the trigger window:
    E = sum_s ( mean_traces(imgA)[s] - mean_traces(imgB)[s] )^2
computed from SINGLE (un-averaged) traces so noise variance is also observable. Run once
per bitstream and compare:

    python fanout_variance_check.py --bitstream ../build/cw305_bnn_3_f16.bit ... --out f16.npz
    python fanout_variance_check.py --bitstream ../build/cw305_bnn_3_f1.bit  ... --out f1.npz
    python fanout_variance_check.py --compare f16.npz f1.npz

Expected if the amp works: E(f16) / E(f1) ~ FANOUT (order-of-magnitude, not exact). If
E(f16) ~ E(f1), the lanes were merged/pruned -> the "16x null" evidence is VOID and the amp
must be rebuilt (distinct lanes) before any bench conclusion stands. HARDWARE-GATED.
"""
import argparse, os
import numpy as np


def capture(args):
    import chipwhisperer as cw
    scope = cw.scope()
    scope.clock.clkgen_freq = args.freq
    scope.default_setup()
    scope.clock.adc_src = "extclk_x4"
    scope.clock.clkgen_freq = args.freq
    EW = 28 + 2 * (args.ksize // 2)
    scope.adc.samples = 4 * EW * EW + 300
    scope.trigger.triggers = "tio4"
    scope.gain.db = args.gain
    target = cw.target(scope, cw.targets.CW305, bsfile=args.bitstream, fpga_id="100t", force=True)
    target.pll.pll_enable_set(True); target.pll.pll_outenable_set(True, 1)
    target.pll.pll_outfreq_set(args.freq, 1)
    scope.clock.reset_adc()

    d = np.load(args.images); xte = d["images"]
    kern = np.load(args.kernels)[args.kernel]
    bits = (kern.flatten() > 0).astype(np.uint8)
    packed = bytearray(); byte = nb = 0
    for b in bits:
        byte = (byte << 1) | int(b); nb += 1
        if nb == 8: packed.append(byte); byte = nb = 0
    if nb: packed.append(byte << (8 - nb))
    target.fpga_write(32, list(packed))

    def grab(img, m):
        target.fpga_write(0, list(img.flatten()))
        rows = []
        for _ in range(m):
            scope.arm(); target.fpga_write(33, [1])
            if scope.capture():
                raise RuntimeError("capture timeout -- check trigger")
            rows.append(scope.get_last_trace())
        return np.asarray(rows, np.float64)

    A = grab(xte[args.imgA].astype(np.uint8), args.m)
    B = grab(xte[args.imgB].astype(np.uint8), args.m)
    scope.dis(); target.dis()
    return A, B


def energy(A, B):
    sig = (A.mean(0) - B.mean(0))
    E = float(np.sum(sig * sig))                     # data-dependent signal energy
    noise = float(0.5 * (A.var(0).mean() + B.var(0).mean()))  # per-sample noise var
    return E, noise


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--compare", nargs=2, help="two saved .npz to compare (E_f16 E_f1)")
    ap.add_argument("--bitstream"); ap.add_argument("--kernels")
    ap.add_argument("--images", default="mnist_test.npz")
    ap.add_argument("--ksize", type=int, choices=[3, 5], default=3)
    ap.add_argument("--kernel", type=int, default=0)
    ap.add_argument("--imgA", type=int, default=0); ap.add_argument("--imgB", type=int, default=1)
    ap.add_argument("--m", type=int, default=200, help="single traces per image")
    ap.add_argument("--freq", type=float, default=5e6); ap.add_argument("--gain", type=float, default=40)
    ap.add_argument("--out", default="fanout_check.npz")
    args = ap.parse_args()

    if args.compare:
        d0, d1 = np.load(args.compare[0]), np.load(args.compare[1])
        E0, N0 = float(d0["E"]), float(d0["noise"]); E1, N1 = float(d1["E"]), float(d1["noise"])
        print(f"{args.compare[0]}: E={E0:.3g} noise={N0:.3g}")
        print(f"{args.compare[1]}: E={E1:.3g} noise={N1:.3g}")
        ratio = E0 / E1 if E1 else float("inf")
        print(f"\ndata-dependent energy ratio E0/E1 = {ratio:.2f}")
        print("  ~FANOUT (>>1): amplifier works, '16x null' evidence stands as measured.")
        print("  ~1: lanes merged/pruned -> '16x null' is VOID; rebuild distinct lanes.")
        return

    if not (args.bitstream and args.kernels):
        raise SystemExit("need --bitstream and --kernels (or --compare)")
    A, B = capture(args)
    E, noise = energy(A, B)
    np.savez(args.out, E=E, noise=noise, A_mean=A.mean(0), B_mean=B.mean(0))
    print(f"saved {args.out}: E={E:.3g} noise={noise:.3g}  (m={args.m} traces/image)")


if __name__ == "__main__":
    main()
