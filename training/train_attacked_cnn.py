#!/usr/bin/env python3
"""
P2 — Train the paper's "attacked" binarized CNN on MNIST.

Reproduces the network of Table 1 in Wei et al., "I Know What You See" (ACSAC'18):
  - 4 weight layers, MNIST 28x28x1
  - 1st layer = convolution, 64 kernels
  - Model 1: 1st-layer kernel 3x3  (paper test acc 99.42%)
  - Model 2: 1st-layer kernel 5x5  (paper test acc 99.27%)
Binarization follows Courbariaux/Hubara BinaryNet (weights & activations in {-1,+1}),
implemented self-contained in bnn_layers.py (Keras 3 compatible).

The first conv layer keeps the REAL pixel input and binary kernels -- exactly the
line-buffer operation the attack targets -- so the exported 64 kernels are the ground
truth the FPGA must load and the side-channel attack profiles.

Usage:
    python train_attacked_cnn.py --ksize 3 --epochs 40
    python train_attacked_cnn.py --ksize 5 --epochs 40

Outputs (under ./artifacts/model_<K>x<K>/):
    weights.weights.h5          trained weights
    layer1_kernels.npy          (64, K, K) int8 in {-1,+1}  -- ground-truth kernels
    layer1_kernels.txt          human-readable
    layer1_kernels_packed.bin   bit-packed for FPGA BRAM (1 bit/weight, MSB-first)
    layer1_bn.npz               layer-1 BatchNorm params (for HW sign threshold, P1)
    metrics.json                test accuracy etc.
"""
import argparse, json, os
import numpy as np


def build_model(ksize):
    import tensorflow as tf
    from bnn_layers import BinaryConv2D, BinaryDense

    inp = tf.keras.Input(shape=(28, 28, 1))

    # Layer 1: 64 conv kernels KxK, REAL input (binarize_input=False), binary weights.
    # padding='same' keeps width 28 -> matches paper "line size = 28".
    x = BinaryConv2D(64, ksize, padding="same", binarize_input=False, name="conv1")(inp)
    x = tf.keras.layers.BatchNormalization(momentum=0.9, name="conv1_bn")(x)

    # Layer 2: 64 conv 3x3 (binary in/out) + pool.
    x = BinaryConv2D(64, 3, padding="same", name="conv2")(x)
    x = tf.keras.layers.MaxPool2D((2, 2), name="pool2")(x)                  # 28 -> 14
    x = tf.keras.layers.BatchNormalization(momentum=0.9, name="conv2_bn")(x)

    x = tf.keras.layers.Flatten()(x)

    # Layer 3: binary dense 1024.
    x = BinaryDense(1024, name="fc3")(x)
    x = tf.keras.layers.BatchNormalization(momentum=0.9, name="fc3_bn")(x)

    # Layer 4: binary dense 10 (output).
    x = BinaryDense(10, name="fc4")(x)
    x = tf.keras.layers.BatchNormalization(momentum=0.9, name="fc4_bn")(x)
    out = tf.keras.layers.Activation("softmax")(x)

    return tf.keras.Model(inp, out)


def load_mnist():
    import tensorflow as tf
    (xtr, ytr), (xte, yte) = tf.keras.datasets.mnist.load_data()
    # Train in [-1,1] for BNN stability. Input scale does NOT affect the binary kernels
    # nor the attack (pixels fed 0..255 to the FPGA at attack time).
    xtr = (xtr.astype("float32") / 127.5 - 1.0)[..., None]
    xte = (xte.astype("float32") / 127.5 - 1.0)[..., None]
    return (xtr, ytr), (xte, yte)


def export_layer1(model, outdir, ksize):
    """Export the 64 binarized first-layer kernels + BN params for the FPGA."""
    w = model.get_layer("conv1").get_weights()[0]      # (K, K, 1, 64) latent
    wb = np.where(w >= 0, 1, -1).astype(np.int8)        # ste_sign: >=0 -> +1
    wb = wb[:, :, 0, :].transpose(2, 0, 1)             # -> (64, K, K)

    np.save(os.path.join(outdir, "layer1_kernels.npy"), wb)
    with open(os.path.join(outdir, "layer1_kernels.txt"), "w") as f:
        for i, k in enumerate(wb):
            f.write(f"# kernel {i}\n")
            for row in k:
                f.write(" ".join(f"{v:+d}" for v in row) + "\n")
            f.write("\n")

    # bit-pack: per kernel, K*K bits row-major MSB-first, 1=+1 / 0=-1.
    packed = bytearray()
    for k in wb:
        bits = (k.flatten() > 0).astype(np.uint8)
        byte = nb = 0
        for b in bits:
            byte = (byte << 1) | int(b); nb += 1
            if nb == 8:
                packed.append(byte); byte = nb = 0
        if nb:
            packed.append(byte << (8 - nb))           # flush, left-aligned
    with open(os.path.join(outdir, "layer1_kernels_packed.bin"), "wb") as f:
        f.write(packed)

    bn = model.get_layer("conv1_bn")
    g, b, m, v = bn.get_weights()
    np.savez(os.path.join(outdir, "layer1_bn.npz"),
             gamma=g, beta=b, mean=m, var=v, epsilon=bn.epsilon)
    print(f"[export] 64 kernels {ksize}x{ksize} -> {outdir} (packed {len(packed)} bytes)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ksize", type=int, choices=[3, 5], required=True,
                    help="first-layer kernel size: 3 (Model 1) or 5 (Model 2)")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--outroot", default="artifacts")
    args = ap.parse_args()

    import tensorflow as tf
    outdir = os.path.join(args.outroot, f"model_{args.ksize}x{args.ksize}")
    os.makedirs(outdir, exist_ok=True)

    (xtr, ytr), (xte, yte) = load_mnist()
    model = build_model(args.ksize)
    model.compile(optimizer=tf.keras.optimizers.Adam(args.lr),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.summary()

    aug = tf.keras.preprocessing.image.ImageDataGenerator(
        width_shift_range=0.1, height_shift_range=0.1)
    sched = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_accuracy", factor=0.5, patience=4, min_lr=1e-5)

    model.fit(aug.flow(xtr, ytr, batch_size=args.batch),
              epochs=args.epochs, validation_data=(xte, yte), callbacks=[sched])

    loss, acc = model.evaluate(xte, yte, verbose=0)
    print(f"[result] Model {args.ksize}x{args.ksize} test acc = {acc*100:.2f}% "
          f"(paper: {'99.42' if args.ksize==3 else '99.27'}%)")

    model.save_weights(os.path.join(outdir, "weights.weights.h5"))
    export_layer1(model, outdir, args.ksize)
    with open(os.path.join(outdir, "metrics.json"), "w") as f:
        json.dump({"ksize": args.ksize, "test_acc": float(acc),
                   "paper_acc": 0.9942 if args.ksize == 3 else 0.9927}, f, indent=2)


if __name__ == "__main__":
    main()
