#!/usr/bin/env python3
"""
Capture on-chip RO-counter sensor traces from the CW305 BNN bitstream.

The FPGA records one 16-bit sensor sample per BNN output pixel cycle. This script reads
those samples over the CW305 USB register interface and writes the same .npz shape that
attack/run_on_hardware.py already consumes.

Example:
  python cw305_tdc_capture.py --bitstream ../build/cw305_bnn_3.bit \
      --kernels ../training/artifacts/model_3x3/layer1_kernels.npy \
      --ksize 3 --n-images 500 --n-kernels 9 --out traces_tdc
"""
import argparse
import os
import time

import numpy as np

LINE = 28
IMG = LINE * LINE

REG_IMAGE = 0
REG_OUTPUT = 16
REG_KERNEL = 32
REG_GO = 33
REG_STATUS = 34
REG_SENSOR_CTRL = 35
REG_SENSOR = 48
DEFAULT_SENSOR_READ_REG = REG_OUTPUT


def pack_kernel(k):
    bits = (k.flatten() > 0).astype(np.uint8)
    out = bytearray()
    byte = nbits = 0
    for bit in bits:
        byte = (byte << 1) | int(bit)
        nbits += 1
        if nbits == 8:
            out.append(byte)
            byte = nbits = 0
    if nbits:
        out.append(byte << (8 - nbits))
    return list(out)


def wait_idle(target, timeout_s=2.0):
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        status = target.fpga_read(REG_STATUS, 1)[0]
        if (status & 1) == 0:
            return
        time.sleep(0.001)
    raise TimeoutError("BNN core did not return idle")


def read_sensor(target, read_reg):
    raw = bytes(target.fpga_read(read_reg, 2 * IMG))
    if len(raw) != 2 * IMG:
        raise RuntimeError(f"sensor read returned {len(raw)} bytes, expected {2 * IMG}")
    return np.frombuffer(raw, dtype="<u2").astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bitstream", required=True)
    ap.add_argument("--kernels", required=True, help="P2 layer1_kernels.npy (64,K,K)")
    ap.add_argument("--ksize", type=int, choices=[3, 5], required=True)
    ap.add_argument("--n-kernels", type=int, default=9)
    ap.add_argument("--n-images", type=int, default=500)
    ap.add_argument("--images", default="mnist_test.npz",
                    help="npz with arrays images(uint8 N,28,28) + labels")
    ap.add_argument("--freq", type=float, default=10e6, help="CW305 target clock Hz")
    ap.add_argument("--out", default="traces_tdc")
    ap.add_argument("--fpga-id", default="100t")
    ap.add_argument("--no-program", action="store_true",
                    help="connect to already-programmed CW305")
    ap.add_argument("--run-wait", type=float, default=0.05,
                    help="seconds to wait after GO before reading sensor RAM")
    ap.add_argument("--sensor-reg", type=int, default=DEFAULT_SENSOR_READ_REG,
                    help="CW305 register to read trace samples from; default 16 for diagnostic builds")
    args = ap.parse_args()

    import chipwhisperer as cw

    os.makedirs(args.out, exist_ok=True)
    kernels = np.load(args.kernels)[:args.n_kernels].astype(np.int8)
    data = np.load(args.images)
    xte, yte = data["images"], data["labels"]
    if args.n_images > len(xte):
        raise ValueError(f"requested {args.n_images} images, only {len(xte)} available")

    bsfile = None if args.no_program else args.bitstream
    target = cw.target(None, cw.targets.CW305, bsfile=bsfile,
                       fpga_id=args.fpga_id, force=not args.no_program)
    try:
        target.pll.pll_enable_set(True)
        target.pll.pll_outenable_set(True, 0)
        target.pll.pll_outenable_set(True, 1)
        target.pll.pll_outfreq_set(args.freq, 1)
        target.fpga_write(REG_SENSOR_CTRL, [1])
        time.sleep(0.1)

        packed = [pack_kernel(k) for k in kernels]
        for n in range(args.n_images):
            outp = os.path.join(args.out, f"img{n:04d}.npz")
            if os.path.exists(outp):
                continue

            img = xte[n].astype(np.uint8)
            target.fpga_write(REG_IMAGE, img.flatten().tolist())

            traces = []
            for kid in range(args.n_kernels):
                target.fpga_write(REG_KERNEL, packed[kid])
                target.fpga_write(REG_GO, [1])
                # The core can finish before a USB status poll ever observes busy.
                time.sleep(args.run_wait)
                wait_idle(target)
                traces.append(read_sensor(target, args.sensor_reg))

            traces = np.asarray(traces, dtype=np.float32)
            np.savez(outp,
                     traces=traces,
                     kernel_ids=np.arange(args.n_kernels),
                     image=img,
                     label=int(yte[n]),
                     samples_per_cycle=1,
                     n_cycles=IMG,
                     trace_source="ro_counter")
            if n % 25 == 0:
                nz = int(np.count_nonzero(traces))
                print(f"captured {n + 1}/{args.n_images} nonzero_sensor={nz}")
    finally:
        target.dis()


if __name__ == "__main__":
    main()
