#!/usr/bin/env python3
"""
Averaging capture of the on-chip RO/TDC voltage sensor (cw305_bnn_ro3.bit).

Sensor is coarse (~11 levels), so per-cycle SNR needs averaging. Fabric-quiescing rule
applied: write image + kernel ONCE, then only GO + read the sensor RAM in the averaging
loop (no register churn mid-average -> the competing USB/reg activity is identical every
run, so its mean cancels and only the conv-driven droop varies). Output .npz matches
attack/run_on_hardware.py (samples_per_cycle=1, n_cycles=784, trace_source='ro_counter').

Run in cwenv:
  python ro_capture.py --n-images 60 --avg 50 --out traces_ro_smoke
"""
import argparse, os, time
import numpy as np

LINE = 28; IMG = LINE * LINE
REG_IMAGE, REG_OUTPUT, REG_KERNEL, REG_GO, REG_STATUS = 0, 16, 32, 33, 34
REG_SENSOR_CTRL = 35


def pack_kernel(k):
    bits = (k.flatten() > 0).astype(np.uint8); out = bytearray(); byte = nb = 0
    for b in bits:
        byte = (byte << 1) | int(b); nb += 1
        if nb == 8: out.append(byte); byte = nb = 0
    if nb: out.append(byte << (8 - nb))
    return list(out)


def wait_idle(target, t=1.0):
    dl = time.time() + t
    while time.time() < dl:
        if (target.fpga_read(REG_STATUS, 1)[0] & 1) == 0:
            return
        time.sleep(0.0005)


def read_sensor(target, reg):
    raw = bytes(target.fpga_read(reg, 2 * IMG))
    return np.frombuffer(raw, dtype="<u2").astype(np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bitstream", default="../build/cw305_bnn_ro3.bit")
    ap.add_argument("--kernels", default="../training/artifacts/model_3x3/layer1_kernels.npy")
    ap.add_argument("--ksize", type=int, default=3)
    ap.add_argument("--n-kernels", type=int, default=9)
    ap.add_argument("--n-images", type=int, default=60)
    ap.add_argument("--avg", type=int, default=50)
    ap.add_argument("--freq", type=float, default=10e6)
    ap.add_argument("--images", default="mnist_test.npz")
    ap.add_argument("--sensor-reg", type=int, default=16)
    ap.add_argument("--run-wait", type=float, default=0.02)
    ap.add_argument("--out", default="traces_ro_smoke")
    args = ap.parse_args()

    import chipwhisperer as cw
    os.makedirs(args.out, exist_ok=True)
    d = np.load(args.images); xte, yte = d["images"], d["labels"]
    kernels = np.load(args.kernels)[:args.n_kernels].astype(np.int8)
    packed = [pack_kernel(k) for k in kernels]

    target = cw.target(None, cw.targets.CW305, bsfile=args.bitstream, fpga_id="100t", force=True)
    try:
        target.pll.pll_enable_set(True)
        target.pll.pll_outenable_set(True, 0); target.pll.pll_outenable_set(True, 1)
        target.pll.pll_outfreq_set(args.freq, 1)
        try:
            target.fpga_write(REG_SENSOR_CTRL, [1]); time.sleep(0.1)
        except Exception:
            pass
        t0 = time.time()
        for n in range(args.n_images):
            outp = os.path.join(args.out, f"img{n:04d}.npz")
            if os.path.exists(outp):
                continue
            img = xte[n].astype(np.uint8)
            target.fpga_write(REG_IMAGE, img.flatten().tolist())   # write image ONCE
            traces = np.zeros((args.n_kernels, IMG), np.float64)
            for kid in range(args.n_kernels):
                target.fpga_write(REG_KERNEL, packed[kid])          # write kernel ONCE
                acc = np.zeros(IMG, np.float64)
                for _ in range(args.avg):                            # quiesced avg loop
                    target.fpga_write(REG_GO, [1])
                    time.sleep(args.run_wait); wait_idle(target)
                    acc += read_sensor(target, args.sensor_reg)
                traces[kid] = acc / args.avg
            np.savez(outp, traces=traces.astype(np.float32),
                     kernel_ids=np.arange(args.n_kernels), image=img, label=int(yte[n]),
                     samples_per_cycle=1, n_cycles=IMG, trace_source="ro_counter",
                     avg=args.avg, freq_hz=float(args.freq))
            if n % 10 == 0:
                rate = (n + 1) / (time.time() - t0)
                print(f"captured {n+1}/{args.n_images}  ({rate:.2f} img/s)  "
                      f"sensor_std={traces.std():.2f}", flush=True)
    finally:
        target.dis()
    print(f"done in {time.time()-t0:.0f}s -> {args.out}")


if __name__ == "__main__":
    main()
