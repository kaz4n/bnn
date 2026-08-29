// P1 -- Testbench for the layer-1 conv unit. Doubles as:
//   (a) Vivado HLS C-sim  (verify RTL behavior), and
//   (b) standalone g++ build (verify datapath + kernel bit-packing WITHOUT Vivado).
//
// Reads data dumped by ../sw_golden/golden_conv.py:
//   data/image.txt          784 ints (pixels 0..255), row-major
//   data/kernel_packed.bin  bit-packed kernel (P2 convention)
//   data/expected.txt       784 ints, numpy 'same' conv accumulator (golden)
// Returns 0 iff hardware output == golden for every pixel.

#include <cstdio>
#include <cstdlib>
#include <vector>
#include "bnn_conv1.h"

void bnn_conv1(const uint8_t img[IMG], const uint8_t packed_kern[(KSIZE*KSIZE+7)/8],
               int16_t out[IMG], uint8_t *leak);

int main(int argc, char** argv) {
    const char* dir = (argc > 1) ? argv[1] : "data";
    char path[512];

    uint8_t img[IMG];
    snprintf(path, sizeof(path), "%s/image.txt", dir);
    FILE* f = fopen(path, "r");
    if (!f) { printf("FAIL: cannot open %s\n", path); return 2; }
    for (int i = 0; i < IMG; i++) { int v; if (fscanf(f, "%d", &v) != 1) { printf("FAIL: image short\n"); return 2;} img[i] = (uint8_t)v; }
    fclose(f);

    const int BPK = (KSIZE*KSIZE + 7) / 8;
    uint8_t packed[BPK];
    snprintf(path, sizeof(path), "%s/kernel_packed.bin", dir);
    f = fopen(path, "rb");
    if (!f) { printf("FAIL: cannot open %s\n", path); return 2; }
    if ((int)fread(packed, 1, BPK, f) != BPK) { printf("FAIL: packed short\n"); return 2; }
    fclose(f);

    std::vector<int> exp(IMG);
    snprintf(path, sizeof(path), "%s/expected.txt", dir);
    f = fopen(path, "r");
    if (!f) { printf("FAIL: cannot open %s\n", path); return 2; }
    for (int i = 0; i < IMG; i++) if (fscanf(f, "%d", &exp[i]) != 1) { printf("FAIL: expected short\n"); return 2; }
    fclose(f);

    int16_t out[IMG];
    uint8_t leak = 0;
    bnn_conv1(img, packed, out, &leak);

    int errors = 0;
    for (int i = 0; i < IMG; i++)
        if ((int)out[i] != exp[i]) {
            if (errors < 10)
                printf("  mismatch @%d (y=%d,x=%d): hw=%d golden=%d\n",
                       i, i/LINE, i%LINE, (int)out[i], exp[i]);
            errors++;
        }

    if (errors == 0) printf("PASS: KSIZE=%d, all %d outputs bit-exact vs golden\n", KSIZE, IMG);
    else             printf("FAIL: KSIZE=%d, %d/%d mismatches\n", KSIZE, errors, IMG);
    return errors ? 1 : 0;
}
