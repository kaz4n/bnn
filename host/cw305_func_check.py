#!/usr/bin/env python3
"""
Hardware functional check (run in cwenv): program the BNN bitstream, push one MNIST
image + one kernel, run the conv, read the feature map back, compare BIT-EXACT to the
numpy golden. Proves the bitstream computes correctly on silicon before any power capture.

    python cw305_func_check.py --bitstream ../build/cw305_bnn_3.bit \
        --kernels ../training/artifacts/model_3x3/layer1_kernels.npy --ksize 3
"""
import argparse, sys, time
import numpy as np

LINE = 28
REG_IMAGE, REG_OUTPUT, REG_KERNEL, REG_GO, REG_STATUS = 0, 16, 32, 33, 34


def conv_same(img, k):
    K = k.shape[0]; p = K // 2
    pad = np.zeros((LINE + 2*p, LINE + 2*p), np.int32)
    pad[p:p+LINE, p:p+LINE] = img.astype(np.int32)
    out = np.zeros((LINE, LINE), np.int32)
    for y in range(LINE):
        for x in range(LINE):
            out[y, x] = int(np.sum(pad[y:y+K, x:x+K] * k.astype(np.int32)))
    return out


def pack(k):
    bits = (k.flatten() > 0).astype(np.uint8)
    out = bytearray(); b = n = 0
    for v in bits:
        b = (b << 1) | int(v); n += 1
        if n == 8: out.append(b); b = n = 0
    if n: out.append(b << (8 - n))
    return list(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bitstream", required=True)
    ap.add_argument("--kernels", required=True)
    ap.add_argument("--ksize", type=int, default=3)
    ap.add_argument("--img", type=int, default=0, help="MNIST test index")
    ap.add_argument("--kidx", type=int, default=0)
    ap.add_argument("--images", default="mnist_test.npz")
    ap.add_argument("--freq", type=float, default=10e6)
    ap.add_argument("--ones", action="store_true", help="self-test: all-1 image, all-+1 kernel")
    args = ap.parse_args()

    import chipwhisperer as cw
    if args.ones:
        img = np.ones((LINE, LINE), np.uint8)            # all pixels = 1
        kern = np.ones((args.ksize, args.ksize), np.int8)  # all +1
    else:
        d = np.load(args.images); imgs = d["images"]
        if imgs.ndim == 4 and imgs.shape[-1] == 1:
            imgs = imgs[..., 0]
        if imgs.ndim != 3 or imgs.shape[1:] != (LINE, LINE):
            raise ValueError(
                f"current CW305 BNN bitstream expects images shaped (N,{LINE},{LINE}); "
                f"got {imgs.shape}"
            )
        kernels = np.load(args.kernels)
        if kernels.ndim != 3 or kernels.shape[1:] != (args.ksize, args.ksize):
            raise ValueError(
                f"current CW305 BNN bitstream expects kernels shaped "
                f"(N,{args.ksize},{args.ksize}); got {kernels.shape}"
            )
        img = imgs[args.img].astype(np.uint8)
        kern = kernels[args.kidx].astype(np.int8)
    golden = conv_same(img, kern).flatten()

    print("[1] program FPGA ...")
    target = cw.target(None, cw.targets.CW305, bsfile=args.bitstream,
                       fpga_id="100t", force=True)
    target.pll.pll_enable_set(True)
    target.pll.pll_outenable_set(True, 1)
    target.pll.pll_outfreq_set(args.freq, 1)

    print("[2] write image + kernel, run conv ...")
    target.fpga_write(REG_IMAGE,  img.flatten().tolist())   # 784 bytes
    target.fpga_write(REG_KERNEL, pack(kern))
    target.fpga_write(REG_GO, [1])
    time.sleep(0.05)
    for _ in range(100):
        if (target.fpga_read(REG_STATUS, 1)[0] & 1) == 0: break
        time.sleep(0.01)

    print("[3] read feature map, compare to golden ...")
    raw = bytes(target.fpga_read(REG_OUTPUT, 2 * LINE * LINE))   # 1568 bytes
    hw = np.frombuffer(raw, dtype="<i2").astype(np.int32)
    print(f"    hw: nonzero={np.count_nonzero(hw)} min={hw.min()} max={hw.max()} | "
          f"golden: nonzero={np.count_nonzero(golden)} min={golden.min()} max={golden.max()}")

    mism = np.nonzero(hw != golden)[0]
    if len(mism) == 0:
        print(f"PASS: all {LINE*LINE} outputs bit-exact (hw == golden). "
              f"range {hw.min()}..{hw.max()}")
    else:
        print(f"FAIL: {len(mism)}/{LINE*LINE} mismatch. first few:")
        for i in mism[:8]:
            print(f"  idx {i} (y={i//LINE},x={i%LINE}): hw={hw[i]} golden={golden[i]}")
    target.dis()
    sys.exit(0 if len(mism) == 0 else 1)


if __name__ == "__main__":
    main()
