#!/usr/bin/env python3
"""Dump MNIST test set to mnist_test.npz so the capture script (run in the ChipWhisperer
cwenv, which has no TensorFlow) needs no TF. Run this ONCE in the project env (has TF):
    python dump_mnist.py
-> mnist_test.npz with images(uint8 N,28,28) + labels."""
import numpy as np
from tensorflow.keras.datasets import mnist
(_, _), (xte, yte) = mnist.load_data()
np.savez("mnist_test.npz", images=xte.astype(np.uint8), labels=yte.astype(np.int64))
print("wrote mnist_test.npz", xte.shape)
