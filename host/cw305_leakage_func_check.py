#!/usr/bin/env python3
"""Functional check for cw305_leakage_top.

Programs the leakage-first 3x3 binary-conv bitstream, writes one binary MNIST
image and one 3x3 binary kernel, then reads back 26x26 signed scores. This proves
the FPGA register map and datapath before any power capture.
"""
import argparse
import os
import time

import numpy as np

LINE = 28
OUT_SIDE = 26
N_OUTPUT = OUT_SIDE * OUT_SIDE
REG_IMAGE = 0
REG_OUTPUT = 16
REG_KERNEL = 32
REG_GO = 33
REG_STATUS = 34


def binarize_image(img, threshold):
    return (np.asarray(img, dtype=np.uint8) > threshold).astype(np.uint8)


def kernel_bits_from_file(path, kidx):
    kernels = np.load(path)
    if kernels.ndim != 3 or kernels.shape[1:] != (3, 3):
        raise ValueError(f"expected kernels shaped (N,3,3), got {kernels.shape}")
    return (kernels[kidx].reshape(-1) > 0).astype(np.uint8)


def onehot_kernel(kidx):
    bits = np.zeros(9, dtype=np.uint8)
    bits[kidx % 9] = 1
    return bits


def pack_kernel_le(bits9):
    value = 0
    for i, bit in enumerate(np.asarray(bits9, dtype=np.uint8).reshape(9)):
        value |= (int(bit) & 1) << i
    return [value & 0xFF, (value >> 8) & 0x01]


def golden_valid_conv(img_bits, kern_bits):
    out = np.zeros(N_OUTPUT, dtype=np.int8)
    idx = 0
    for y in range(OUT_SIDE):
        for x in range(OUT_SIDE):
            patch = img_bits[y:y + 3, x:x + 3].reshape(-1)
            matches = int(np.count_nonzero(patch == kern_bits))
            out[idx] = np.int8(2 * matches - 9)
            idx += 1
    return out


def wait_idle(target, timeout_s):
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        status = target.fpga_read(REG_STATUS, 1)[0]
        if (status & 1) == 0:
            return
        time.sleep(0.002)
    raise TimeoutError("FPGA did not go idle")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bitstream", default="../build/cw305_leakage_d8_l63.bit")
    ap.add_argument("--images", default="mnist_test.npz")
    ap.add_argument("--kernels", default="../training/artifacts/model_3x3/layer1_kernels.npy")
    ap.add_argument("--probe-kernels", choices=["onehot", "file"], default="onehot")
    ap.add_argument("--img", type=int, default=0)
    ap.add_argument("--kidx", type=int, default=0)
    ap.add_argument("--threshold", type=int, default=127)
    ap.add_argument("--freq", type=float, default=5e6)
    ap.add_argument("--fpga-id", default="100t")
    ap.add_argument("--no-program", action="store_true")
    ap.add_argument("--ones", action="store_true")
    args = ap.parse_args()

    import chipwhisperer as cw

    if args.ones:
        img_bits = np.ones((LINE, LINE), dtype=np.uint8)
        kern_bits = np.ones(9, dtype=np.uint8)
    else:
        d = np.load(args.images)
        imgs = d["images"]
        if imgs.ndim == 4 and imgs.shape[-1] == 1:
            imgs = imgs[..., 0]
        if imgs.ndim != 3 or imgs.shape[1:] != (LINE, LINE):
            raise ValueError(f"expected images shaped (N,28,28), got {imgs.shape}")
        img_bits = binarize_image(imgs[args.img], args.threshold)
        if args.probe_kernels == "onehot":
            kern_bits = onehot_kernel(args.kidx)
        else:
            kern_bits = kernel_bits_from_file(args.kernels, args.kidx)

    bsfile = None if args.no_program else os.path.abspath(args.bitstream)
    target = cw.target(None, cw.targets.CW305, bsfile=bsfile,
                       fpga_id=args.fpga_id, force=not args.no_program)
    target.pll.pll_enable_set(True)
    target.pll.pll_outenable_set(True, 1)
    target.pll.pll_outfreq_set(args.freq, 1)

    target.fpga_write(REG_IMAGE, img_bits.reshape(-1).astype(np.uint8).tolist())
    target.fpga_write(REG_KERNEL, pack_kernel_le(kern_bits))
    target.fpga_write(REG_GO, [1])
    wait_idle(target, 1.0)

    raw = bytes(target.fpga_read(REG_OUTPUT, N_OUTPUT))
    hw = np.frombuffer(raw, dtype=np.int8)
    golden = golden_valid_conv(img_bits, kern_bits)
    mism = np.nonzero(hw != golden)[0]

    print(f"bitstream={bsfile or '(already programmed)'}")
    print(f"freq={args.freq:g} image_bits={int(img_bits.sum())} "
          f"kernel_bits={''.join(str(int(x)) for x in kern_bits)}")
    print(f"hw range={int(hw.min())}..{int(hw.max())} "
          f"golden range={int(golden.min())}..{int(golden.max())}")
    if len(mism):
        print(f"FAIL: {len(mism)}/{N_OUTPUT} mismatches")
        for i in mism[:8]:
            print(f"  out[{i}] y={i // OUT_SIDE} x={i % OUT_SIDE}: "
                  f"hw={int(hw[i])} golden={int(golden[i])}")
        target.dis()
        raise SystemExit(1)

    print(f"PASS: {N_OUTPUT} outputs bit-exact")
    target.dis()


if __name__ == "__main__":
    main()
