// P1 -- Layer-1 binarized conv unit (the side-channel attack target).
//
// Reproduces the bnn-fpga line-buffer convolution for the paper's first layer:
//   input  : 28x28x1, REAL pixels (paper feeds 0..255), 8-bit unsigned
//   kernel : KxK binary {-1,+1}, ONE output feature map per invocation
//            (the attack captures one power trace per kernel; 64 kernels total)
//   output : 28x28 'same' convolution accumulator (int16), border zero-padded
//
// One pixel is consumed per clock cycle (line-buffer behavior) -- this is what makes
// the per-cycle <-> pixel-window mapping the attack relies on hold. See bnn_conv1.cpp
// for the synthesizable top with HLS pragmas; this header holds the algorithm so the
// exact same code is used by HLS C-sim and the standalone g++ testbench (bit-exact).

#ifndef BNN_CONV1_H
#define BNN_CONV1_H

#include <stdint.h>

#ifndef KSIZE
#define KSIZE 3            // 3 = Model 1, 5 = Model 2 (set by build)
#endif

#define LINE 28            // paper "line size = 28"
#define IMG  (LINE*LINE)   // 784
#define PAD  (KSIZE/2)     // 'same' padding

// kern[]: KSIZE*KSIZE values, each -1 or +1, row-major.
// img[]:  LINE*LINE pixels, row-major.
// out[]:  LINE*LINE accumulators, row-major.
static inline void bnn_conv1_core(const uint8_t img[IMG],
                                  const int8_t  kern[KSIZE*KSIZE],
                                  int16_t       out[IMG])
{
    for (int y = 0; y < LINE; y++) {
        for (int x = 0; x < LINE; x++) {
            int16_t acc = 0;
            for (int i = 0; i < KSIZE; i++) {
                for (int j = 0; j < KSIZE; j++) {
                    int yy = y + i - PAD;
                    int xx = x + j - PAD;
                    uint8_t p = (yy >= 0 && yy < LINE && xx >= 0 && xx < LINE)
                                ? img[yy*LINE + xx] : 0;   // zero-pad border
                    int8_t  w = kern[i*KSIZE + j];          // -1 or +1
                    acc += (int16_t)(w == 1 ? (int16_t)p : -(int16_t)p);
                }
            }
            out[y*LINE + x] = acc;
        }
    }
}

// Unpack a bit-packed kernel produced by P2 (training/export_layer1):
//   KSIZE*KSIZE bits, row-major, MSB-first, 1 -> +1, 0 -> -1.
// bytes_per_kernel = ceil(KSIZE*KSIZE / 8).
static inline void bnn_unpack_kernel(const uint8_t* packed, int8_t kern[KSIZE*KSIZE])
{
    for (int b = 0; b < KSIZE*KSIZE; b++) {
        int byte = b / 8;
        int bit  = 7 - (b % 8);                 // MSB-first
        int v    = (packed[byte] >> bit) & 1;
        kern[b]  = v ? 1 : -1;
    }
}

#endif // BNN_CONV1_H
