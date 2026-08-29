// P1 -- Layer-1 binarized conv: shift-register line buffer + FANOUT MAC leakage
// amplifier on the KERNEL-DEPENDENT datapath (paper-faithful, scaled for a less-sensitive
// bench).
//
// Why not just replicate the pixel shift registers?  The paper's two attacks read
// DIFFERENT things:
//   * S6 background detection reads per-cycle switching MAGNITUDE (a uniform window causes
//     no product switching -> low power). Pixel-shift activity carries this.
//   * S7 power template reads the ACROSS-KERNEL power-feature-vector rho: for one cycle it
//     compares power under 9 different kernels and infers the window pixels. That signal
//     lives ONLY in the kernel-dependent part of the datapath (products g=w*x and the
//     adder tree). Raw pixel shifting is kernel-INDEPENDENT, so amplifying it does NOTHING
//     for S7 -- the across-kernel variance stays at the noise floor and the template attack
//     fails on hardware even though it "works" against a shift-register-only leakage sim.
//
// So the amplifier must ride the KERNEL-DEPENDENT datapath. FANOUT MAC lanes each compute
// products of the REAL kernel with the pixel XOR'd by a distinct per-lane constant c_f:
//
//     g_f[m] = kern[m] * (pixel[m] ^ c_f)          c_0 = 0  => lane 0 is the TRUE conv
//
// The XOR keeps the lanes logically DISTINCT so Vivado cannot merge them (dedup would kill
// the gain). Crucially the XOR only changes the product MAGNITUDE; the sign is kern[m],
// applied identically in every lane -- so summing the FANOUT lanes AMPLIFIES the
// kernel-dependence (~FANOUT x) instead of averaging it away. (A per-lane SIGN flip would
// have cancelled it: balanced +/- over lanes erases the kernel sign. Magnitude XOR does
// not.) Lane 0 (c_0=0) reproduces the bit-exact conv output. The product/accumulator
// registers pg/pacc are the flip-flops that switch each cycle; a `leak` sink reads their
// final state so Vivado retains them.
//
// trace_sim.per_cycle_power models THIS mechanism (same FANOUT, same c_f, same XOR-in-MAC),
// so the software attack validates the leakage the hardware actually produces.

#include "bnn_conv1.h"

#define PAD (KSIZE/2)
#define EW  (LINE + 2*PAD)
#define EH  (LINE + 2*PAD)
#ifndef FANOUT
#define FANOUT 16          // number of parallel kernel-dependent MAC lanes (leakage gain)
#endif
// per-lane XOR constant applied to the MAC pixel input; c_f = f*LANE_C (mod 256), so
// c_0 = 0 (lane 0 == true conv) and the lanes are pairwise distinct for f < 256.
#define LANE_C 0x35

void bnn_conv1(const uint8_t img[IMG],
               const uint8_t packed_kern[(KSIZE*KSIZE+7)/8],
               int16_t       out[IMG],
               uint8_t      *leak)            // sink: forces the FANOUT lanes to be kept
{
#pragma HLS INTERFACE bram   port=img
#pragma HLS INTERFACE bram   port=packed_kern
#pragma HLS INTERFACE bram   port=out
#pragma HLS INTERFACE ap_none port=leak
#pragma HLS INTERFACE ap_ctrl_hs port=return

    int8_t kern[KSIZE*KSIZE];
#pragma HLS ARRAY_PARTITION variable=kern complete dim=1
    bnn_unpack_kernel(packed_kern, kern);

    // ONE real line buffer + window (shift registers carry the true pixel stream; this is
    // the S6 pixel-switching path and feeds lane 0's bit-exact output).
    static uint8_t lb[KSIZE-1][EW];
#pragma HLS ARRAY_PARTITION variable=lb complete dim=0
    uint8_t win[KSIZE][KSIZE];
#pragma HLS ARRAY_PARTITION variable=win complete dim=0

    // FANOUT kernel-dependent MAC lanes: pg = per-lane product regs, pacc = per-lane
    // accumulator regs. These flip-flops switch every cycle and carry the amplified,
    // kernel-dependent activity the S7 template attack reads.
    static int16_t pg[FANOUT][KSIZE*KSIZE];
#pragma HLS ARRAY_PARTITION variable=pg complete dim=0
    static int16_t pacc[FANOUT];
#pragma HLS ARRAY_PARTITION variable=pacc complete dim=0

INIT: for (int i = 0; i < KSIZE-1; i++)
        for (int c = 0; c < EW; c++) lb[i][c] = 0;
    for (int i = 0; i < KSIZE; i++)
        for (int j = 0; j < KSIZE; j++) win[i][j] = 0;
    for (int f = 0; f < FANOUT; f++) {
        for (int m = 0; m < KSIZE*KSIZE; m++) pg[f][m] = 0;
        pacc[f] = 0;
    }

ROW: for (int er = 0; er < EH; er++) {
COL:    for (int ec = 0; ec < EW; ec++) {
#pragma HLS PIPELINE II=1
            uint8_t in = (er >= PAD && er < PAD+LINE && ec >= PAD && ec < PAD+LINE)
                         ? img[(er-PAD)*LINE + (ec-PAD)] : (uint8_t)0;

            // advance the single real line buffer + window with the true pixel stream
            uint8_t newcol[KSIZE];
            for (int i = 0; i < KSIZE-1; i++) newcol[i] = lb[i][EW-1];
            newcol[KSIZE-1] = in;
            for (int i = 0; i < KSIZE-1; i++) {
                uint8_t feed = (i == KSIZE-2) ? in : lb[i+1][EW-1];
                for (int c = EW-1; c > 0; c--) lb[i][c] = lb[i][c-1];
                lb[i][0] = feed;
            }
            for (int i = 0; i < KSIZE; i++) {
                for (int j = 0; j < KSIZE-1; j++) win[i][j] = win[i][j+1];
                win[i][KSIZE-1] = newcol[i];
            }

            // FANOUT kernel-dependent MAC lanes over the shared real window
        FAN: for (int f = 0; f < FANOUT; f++) {
#pragma HLS UNROLL
                uint8_t cf = (uint8_t)(f * LANE_C);        // c_0 = 0 => true conv
                int16_t s = 0;
                for (int i = 0; i < KSIZE; i++) {
                    for (int j = 0; j < KSIZE; j++) {
                        uint8_t px = (uint8_t)(win[i][j] ^ cf);   // magnitude perturb only
                        int16_t prod = (kern[i*KSIZE+j] == 1) ? (int16_t)px
                                                              : (int16_t)(-(int16_t)px);
                        pg[f][i*KSIZE+j] = prod;                  // kernel sign kept in all lanes
                        s += prod;
                    }
                }
                pacc[f] = s;
            }

            if (er >= KSIZE-1 && ec >= KSIZE-1) {
                int oy = er - (KSIZE-1);
                int ox = ec - (KSIZE-1);
                out[oy*LINE + ox] = pacc[0];   // lane 0 (c_0=0) == bit-exact conv output
            }
        }
    }
    // sink reads the FINAL state of every amplifier + shift register -> retention
    uint8_t s = 0;
SINK_G: for (int f = 0; f < FANOUT; f++)
        for (int m = 0; m < KSIZE*KSIZE; m++) s ^= (uint8_t)pg[f][m];
SINK_A: for (int f = 0; f < FANOUT; f++) s ^= (uint8_t)pacc[f];
SINK_W: for (int i = 0; i < KSIZE; i++)
        for (int j = 0; j < KSIZE; j++) s ^= win[i][j];
SINK_L: for (int i = 0; i < KSIZE-1; i++)
        for (int c = 0; c < EW; c++) s ^= lb[i][c];
    *leak = s;
}
