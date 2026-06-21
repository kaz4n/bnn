"""
Self-contained BinaryNet binarization (Courbariaux/Hubara) for TF 2.x / Keras 3.

Drop-in for larq (which is incompatible with Keras 3). The math is identical to
BinaryNet: weights & activations -> {-1,+1} via sign, gradients via the
straight-through estimator (STE) clipped to |x|<=1, and latent weights clipped to
[-1,1].
"""
import tensorflow as tf


@tf.custom_gradient
def ste_sign(x):
    """Forward: sign(x) with 0 -> +1. Backward: straight-through, clipped to |x|<=1."""
    y = tf.sign(x)
    y = tf.where(tf.equal(y, 0.0), tf.ones_like(y), y)

    def grad(dy):
        return dy * tf.cast(tf.abs(x) <= 1.0, x.dtype)

    return y, grad


class ClipWeights(tf.keras.constraints.Constraint):
    """Clip latent weights to [-1, 1] (BinaryNet)."""
    def __call__(self, w):
        return tf.clip_by_value(w, -1.0, 1.0)


_clip = ClipWeights()


class BinaryConv2D(tf.keras.layers.Layer):
    """Conv2D with binary {-1,+1} kernel; optionally binarized input activations."""

    def __init__(self, filters, ksize, padding="same", binarize_input=True, **kw):
        super().__init__(**kw)
        self.filters = filters
        self.ksize = ksize
        self.padding = padding.upper()
        self.binarize_input = binarize_input

    def build(self, shape):
        cin = int(shape[-1])
        self.kernel = self.add_weight(
            name="kernel", shape=(self.ksize, self.ksize, cin, self.filters),
            initializer="glorot_uniform", constraint=_clip, trainable=True)

    def call(self, x):
        if self.binarize_input:
            x = ste_sign(x)
        kb = ste_sign(self.kernel)
        return tf.nn.conv2d(x, kb, strides=1, padding=self.padding)

    def get_config(self):
        c = super().get_config()
        c.update(filters=self.filters, ksize=self.ksize,
                 padding=self.padding, binarize_input=self.binarize_input)
        return c


class BinaryDense(tf.keras.layers.Layer):
    """Dense with binary {-1,+1} kernel; optionally binarized input activations."""

    def __init__(self, units, binarize_input=True, **kw):
        super().__init__(**kw)
        self.units = units
        self.binarize_input = binarize_input

    def build(self, shape):
        self.kernel = self.add_weight(
            name="kernel", shape=(int(shape[-1]), self.units),
            initializer="glorot_uniform", constraint=_clip, trainable=True)

    def call(self, x):
        if self.binarize_input:
            x = ste_sign(x)
        return tf.matmul(x, ste_sign(self.kernel))

    def get_config(self):
        c = super().get_config()
        c.update(units=self.units, binarize_input=self.binarize_input)
        return c
