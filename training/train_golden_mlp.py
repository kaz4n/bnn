#!/usr/bin/env python3
"""
P2 — Train the "golden reference" MNIST classifier used to score recovered images.

Paper §6.3: recognition accuracy is measured by feeding each recovered image to
"a multi-layer perceptron network [12] with an accuracy of 99.2% as a golden
reference". Ref [12] is BinaryNet (Hubara/Courbariaux); its MNIST model is an MLP
(3 hidden layers, 4096 binary units). This is NOT the attacked network -- it only
scores how recognizable a recovered image is (the recognition-accuracy metric in
§6/§7).

Usage:
    python train_golden_mlp.py --epochs 50

Output:
    artifacts/golden_mlp/weights.weights.h5
    artifacts/golden_mlp/metrics.json
The attack scripts (P3) load this to compute recognition accuracy.
"""
import argparse, json, os


def build_mlp():
    import tensorflow as tf
    from bnn_layers import BinaryDense
    inp = tf.keras.Input(shape=(28, 28, 1))
    x = tf.keras.layers.Flatten()(inp)
    for i in range(3):                                   # 3 hidden binary layers, 4096
        x = tf.keras.layers.Dropout(0.2)(x)
        # first hidden layer takes real pixels (binarize_input=False)
        x = BinaryDense(4096, binarize_input=(i > 0), name=f"fc{i}")(x)
        x = tf.keras.layers.BatchNormalization(momentum=0.9, name=f"fc{i}_bn")(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = BinaryDense(10, name="fc_out")(x)
    x = tf.keras.layers.BatchNormalization(momentum=0.9, name="out_bn")(x)
    out = tf.keras.layers.Activation("softmax")(x)
    return tf.keras.Model(inp, out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--outroot", default="artifacts")
    ap.add_argument("--dataset", choices=["mnist", "fashion"], default="mnist",
                    help="which 28x28 set to train the recogniser on; the "
                         "Fashion weights go to a separate artifact directory "
                         "so the MNIST recogniser is never overwritten")
    args = ap.parse_args()

    import tensorflow as tf
    from train_attacked_cnn import load_mnist
    name = "golden_mlp" if args.dataset == "mnist" else f"golden_mlp_{args.dataset}"
    outdir = os.path.join(args.outroot, name)
    os.makedirs(outdir, exist_ok=True)

    (xtr, ytr), (xte, yte) = load_mnist(args.dataset)
    model = build_mlp()
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.summary()
    sched = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_accuracy", factor=0.5, patience=4, min_lr=1e-5)
    model.fit(xtr, ytr, epochs=args.epochs, batch_size=args.batch,
              validation_data=(xte, yte), callbacks=[sched])

    acc = model.evaluate(xte, yte, verbose=0)[1]
    print(f"[result] golden MLP test acc = {acc*100:.2f}% (paper: 99.2%)")
    model.save_weights(os.path.join(outdir, "weights.weights.h5"))
    with open(os.path.join(outdir, "metrics.json"), "w") as f:
        json.dump({"test_acc": float(acc), "paper_acc": 0.992}, f, indent=2)


if __name__ == "__main__":
    main()
