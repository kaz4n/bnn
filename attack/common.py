"""
P3 -- shared helpers: scan order, line-buffer window/related-pixel mapping, MNIST.

The conv unit emits ONE output pixel per clock cycle (line buffer), scanning the 28x28
'same'-padded feature map row-major. So cycle c <-> output pixel (y,x)=divmod(c,28).
This cycle<->pixel mapping is the backbone of both attacks (paper S6/S7).
"""
import numpy as np

LINE = 28
NCYC = LINE * LINE          # 784 cycles per feature map


def cyc_to_yx(c):
    return c // LINE, c % LINE


def window_pixels(img, y, x, K):
    """K x K window of img centered at (y,x), zero-padded ('same'). Returns flat (K*K,)."""
    p = K // 2
    out = np.zeros(K * K, dtype=np.int32)
    for i in range(K):
        for j in range(K):
            yy, xx = y + i - p, x + j - p
            if 0 <= yy < LINE and 0 <= xx < LINE:
                out[i * K + j] = img[yy, xx]
    return out


def related_pixels(img, c, K):
    """
    Paper S7.2: the related pixels for cycle c are the union of the convolution window
    at cycle c-1 and c (the window shifts one column). Size K*(K+1).
    Returns (positions list of (y,x), values np.array) for the union region, ordered
    deterministically so the same region maps to the same vector everywhere.
    """
    y, x = cyc_to_yx(c)
    p = K // 2
    # union columns: x-1-p .. x+p  -> (K+1) columns; rows y-p .. y+p -> K rows
    positions, values = [], []
    for i in range(K):                         # rows
        for j in range(K + 1):                 # cols (one extra for the shift)
            yy = y + i - p
            xx = (x - 1) + j - p
            positions.append((yy, xx))
            if 0 <= yy < LINE and 0 <= xx < LINE:
                values.append(int(img[yy, xx]))
            else:
                values.append(0)
    return positions, np.array(values, dtype=np.int32)


def load_mnist_test():
    from tensorflow.keras.datasets import mnist
    (_, _), (xte, yte) = mnist.load_data()
    return xte.astype(np.uint8), yte.astype(np.int64)
