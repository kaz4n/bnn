#!/usr/bin/env python3
"""
P1 -- numpy golden for the layer-1 conv unit + test-vector dumper.

Computes the reference 'same' convolution accumulator the HLS/RTL conv unit must match
bit-for-bit, and dumps vectors for hls/bnn_conv1_tb.cpp. Verifies BOTH the datapath and
the P2 kernel bit-packing convention end-to-end.

Usage:
  # use a trained kernel from P2 (kernel index --kidx of Model 1):
  python golden_conv.py --ksize 3 --kernels ../training/artifacts/model_3x3/layer1_kernels.npy --kidx 0
  # or self-contained random test (no trained weights needed):
  python golden_conv.py --ksize 3 --random --seed 0
Outputs to ./data/: image.txt, kernel_packed.bin, kernel.txt, expected.txt
"""
import argparse, os
import numpy as np

LINE = 28


def pack_kernel(k):
    """k: (K,K) in {-1,+1} -> bytes, K*K bits row-major MSB-first, 1=+1/0=-1."""
    bits = (k.flatten() > 0).astype(np.uint8)
    out = bytearray()
    byte = nb = 0
    for b in bits:
        byte = (byte << 1) | int(b); nb += 1
        if nb == 8:
            out.append(byte); byte = nb = 0
    if nb:
        out.append(byte << (8 - nb))
    return bytes(out)


def conv_same(img, k):
    """Integer 'same' conv, zero-padded. img uint8 (28,28), k (K,K) in {-1,1}."""
    K = k.shape[0]; p = K // 2
    padded = np.zeros((LINE + 2*p, LINE + 2*p), dtype=np.int32)
    padded[p:p+LINE, p:p+LINE] = img.astype(np.int32)
    out = np.zeros((LINE, LINE), dtype=np.int32)
    for y in range(LINE):
        for x in range(LINE):
            win = padded[y:y+K, x:x+K]
            out[y, x] = int(np.sum(win * k.astype(np.int32)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ksize", type=int, choices=[3, 5], required=True)
    ap.add_argument("--kernels", help="P2 layer1_kernels.npy")
    ap.add_argument("--kidx", type=int, default=0)
    ap.add_argument("--image", help="optional .npy MNIST image (28,28) uint8")
    ap.add_argument("--random", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="data")
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    os.makedirs(args.out, exist_ok=True)

    # kernel
    if args.random or not args.kernels:
        k = rng.choice([-1, 1], size=(args.ksize, args.ksize)).astype(np.int8)
    else:
        allk = np.load(args.kernels)
        assert allk.shape[1] == args.ksize, "kernel size mismatch"
        k = allk[args.kidx].astype(np.int8)

    # image (0..255)
    if args.image:
        img = np.load(args.image).astype(np.uint8)
    elif args.random or not args.kernels:
        img = rng.integers(0, 256, size=(LINE, LINE), dtype=np.uint8)
    else:
        import tensorflow as tf
        (_, _), (xte, _) = tf.keras.datasets.mnist.load_data()
        img = xte[0].astype(np.uint8)

    exp = conv_same(img, k)

    np.savetxt(os.path.join(args.out, "image.txt"), img.flatten(), fmt="%d")
    np.savetxt(os.path.join(args.out, "expected.txt"), exp.flatten(), fmt="%d")
    with open(os.path.join(args.out, "kernel.txt"), "w") as f:
        for row in k:
            f.write(" ".join(f"{v:+d}" for v in row) + "\n")
    with open(os.path.join(args.out, "kernel_packed.bin"), "wb") as f:
        f.write(pack_kernel(k))
    print(f"[golden] KSIZE={args.ksize} wrote vectors to {args.out}/ "
          f"(acc range {exp.min()}..{exp.max()})")


if __name__ == "__main__":
    main()
