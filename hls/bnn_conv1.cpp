// P1 -- Synthesizable HLS top for the layer-1 binarized conv unit.
//
// Vivado HLS 2016.4 flow (see ../build/run_hls.tcl):
//   csim  -> verify bit-exact vs numpy golden (bnn_conv1_tb.cpp)
//   csynth-> RTL
//   export_design (ip_catalog) -> Vivado IP wrapped by ../cw305/cw305_bnn_top.v
//
// NOTE: stock bnn-fpga uses `#pragma SDS ...` (SDSoC). You have plain Vivado HLS, so we
// use plain HLS INTERFACE pragmas instead. The trigger is asserted in the Verilog
// wrapper from the ap_start..ap_done handshake (NOT here) so it spans exactly the conv.

#include "bnn_conv1.h"

// BRAM-mapped I/O. img/out as BRAM ports, kern packed in a small BRAM, ap_ctrl_hs so
// the wrapper sees ap_start/ap_done (used to drive the CW trigger).
void bnn_conv1(const uint8_t img[IMG],
               const uint8_t packed_kern[(KSIZE*KSIZE+7)/8],
               int16_t       out[IMG])
{
#pragma HLS INTERFACE bram   port=img
#pragma HLS INTERFACE bram   port=packed_kern
#pragma HLS INTERFACE bram   port=out
#pragma HLS INTERFACE ap_ctrl_hs port=return

    int8_t kern[KSIZE*KSIZE];
#pragma HLS ARRAY_PARTITION variable=kern complete
    bnn_unpack_kernel(packed_kern, kern);

    bnn_conv1_core(img, kern, out);
}
