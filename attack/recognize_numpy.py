#!/usr/bin/env python3
"""NumPy forward pass of the golden BinaryNet MLP used for recognition accuracy.

The training-time model lives in TensorFlow (`training/train_golden_mlp.py`),
but the capture/attack environment has no TensorFlow.  The network is small and
deterministic at inference time, so the forward pass is reimplemented here from
the saved Keras weights:

    flatten -> 3 x [BinaryDense(4096) + BatchNorm] -> BinaryDense(10) + BatchNorm

BinaryDense uses sign(W) as the kernel and sign(x) on its input for every layer
except the first, matching `training/bnn_layers.py`.
"""
import argparse
import os

import numpy as np

DEFAULT_WEIGHTS = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "training", "artifacts",
    "golden_mlp", "weights.weights.h5")
BN_EPS = 1e-3


def sign_pm1(x):
    """sign(x) with 0 mapped to +1, as in ste_sign."""
    return np.where(x >= 0, 1.0, -1.0).astype(np.float32)


class GoldenMLP:
    def __init__(self, weights_path=None):
        """Load the binarized recogniser.

        The weights default to the MNIST digit model, but every scoring path
        below calls GoldenMLP() with no argument, so a Fashion-MNIST run would
        silently score clothing with a digit classifier and report a number that
        looks plausible and means nothing. BNN_RECOGNIZER_WEIGHTS overrides the
        default so the dataset can be switched without touching five call sites.
        """
        if weights_path is None:
            weights_path = os.environ.get("BNN_RECOGNIZER_WEIGHTS", DEFAULT_WEIGHTS)
        import h5py

        with h5py.File(os.path.abspath(weights_path), "r") as f:
            layers = f["layers"]
            self.kernels = [
                sign_pm1(np.asarray(layers[name]["vars"]["0"], dtype=np.float32))
                for name in ("binary_dense", "binary_dense_1",
                             "binary_dense_2", "binary_dense_3")
            ]
            self.bns = []
            for name in ("batch_normalization", "batch_normalization_1",
                         "batch_normalization_2", "batch_normalization_3"):
                v = layers[name]["vars"]
                self.bns.append(tuple(
                    np.asarray(v[str(i)], dtype=np.float32) for i in range(4)))

    @staticmethod
    def _bn(x, params):
        gamma, beta, mean, var = params
        return gamma * (x - mean) / np.sqrt(var + BN_EPS) + beta

    def logits(self, images01):
        """images01: (N,28,28) in [0,1] scale (binary or grayscale/255)."""
        x = np.asarray(images01, dtype=np.float32).reshape(len(images01), -1)
        x = (x * 255.0) / 127.5 - 1.0
        for i in range(4):
            if i > 0:
                x = sign_pm1(x)
            x = x @ self.kernels[i]
            x = self._bn(x, self.bns[i])
        return x

    def predict(self, images01, batch=256):
        images01 = np.asarray(images01, dtype=np.float32)
        out = np.zeros(len(images01), dtype=np.int64)
        for i in range(0, len(images01), batch):
            out[i:i + batch] = self.logits(images01[i:i + batch]).argmax(axis=1)
        return out


def main():
    ap = argparse.ArgumentParser(description="self-test on MNIST test images")
    ap.add_argument("--images", default="../host/mnist_test.npz")
    ap.add_argument("--weights", default=DEFAULT_WEIGHTS)
    ap.add_argument("--count", type=int, default=2000)
    ap.add_argument("--threshold", type=int, default=127)
    args = ap.parse_args()

    d = np.load(args.images)
    imgs = d["images"][:args.count]
    labels = d["labels"][:args.count]
    model = GoldenMLP(args.weights)
    gray = model.predict(imgs.astype(np.float32) / 255.0)
    binar = model.predict((imgs > args.threshold).astype(np.float32))
    print(f"grayscale accuracy  : {np.mean(gray == labels):.4f}")
    print(f"binarized accuracy  : {np.mean(binar == labels):.4f}")


if __name__ == "__main__":
    main()
