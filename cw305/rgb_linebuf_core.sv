`timescale 1ns/1ps
`default_nettype none

// RGB 3x3 first-layer convolution with a LINE BUFFER, for CW305/CW-Lite research.
//
// SUPERSEDES cw305/rgb_leakage_core.sv (kept only for reference; do not build it).
// That version held the image in a register-file array and read 27
// taps at random offsets. That did not fit: out-of-context synthesis reported 73,174 LUTs
// (115% of an xc7a100t) with LUT-as-Memory at 1.7% and F7 muxes at 93% -- Vivado inferred
// no distributed RAM and built 27 wide multiplexers out of logic. An array with nine
// asynchronous, independently-addressed read ports does not infer LUTRAM at this width.
//
// A line buffer is both the fix and the more faithful structure. Wei et al. are explicit
// that this is what the attack targets: "the actual attack target is the structure of
// line buffer where we exploit the power consumption with the sliding convolutional
// window". Storage becomes two image rows plus the 3x3 window in shift registers -- about
// 2*IMG_SIDE + 9 pixel registers -- and the image itself moves to block RAM, streamed one
// RGB pixel per window advance.
//
// PIXEL FORMAT
// ------------
// One 24-bit word per spatial position, {B, G, R} with R in bits [7:0]. Packing the three
// channels into one word is what lets a single BRAM port feed the whole window; planar
// storage would need three concurrent reads.
//
// SCAN GEOMETRY
// -------------
// Every one of IMG_SIDE*IMG_SIDE positions is streamed in raster order, because the line
// buffer must see them all to stay coherent. Only positions with row >= 2 and col >= 2
// produce an output, giving the usual (IMG_SIDE-2)^2 'valid' convolution results.
//
// SEGMENT-FRAMED TRIGGER
// ----------------------
// A full 32x32 scan at DWELL=8 is 1024*8*4 = 32,768 ADC samples, and 98,304 in serial
// mode, against a ChipWhisperer-Lite limit near 24,573. Rather than shrink the image or
// collapse DWELL to 2 -- which would destroy the settling structure the measured
// featurization depends on -- the core always scans the whole image and asserts
// tio_trigger only while the position index lies in [seg_start, seg_start+seg_len).
// The host captures a full image in a few segments and concatenates them. Geometry and
// DWELL are preserved; only acquisition time grows.
//
// DATAFLOW MODES are unchanged from rgb_leakage_core.sv and still runtime-selected:
//   0 serial   -- one dwell period per channel, three per position; each channel gets its
//                 own settled window, which is the response to the measured inter-window
//                 smear.
//   1 parallel -- one dwell period; all 27 products registered.
//   2 summed   -- one dwell period; product registers held, so only the accumulator
//                 toggles. Worst case for the attacker.
// The convolution RESULT is identical in all three. Only the registered state differs.
// USE_DSP is disabled: the adder tree must stay in fabric. A DSP48 has a different power
// signature from a LUT adder, so mapping the MAC into DSPs would change the very quantity
// this design exists to measure, and would diverge from the grey baseline it is compared
// against.
(* use_dsp48 = "no" *)
module rgb_linebuf_core #(
    parameter integer DWELL    = 8,
    parameter integer IMG_SIDE = 32
)(
    input  wire                 clk,
    input  wire                 reset,
    input  wire                 start,
    input  wire [1:0]           mode,
    input  wire [26:0]          kernel_bits,   // one +/-1 weight per (channel, tap)
    // Segment framing for the trigger. seg_len = 0 means "frame the whole scan".
    input  wire [15:0]          seg_start,
    input  wire [15:0]          seg_len,
    // Streaming pixel port. pix_addr is COMBINATIONAL and pix_data is the registered
    // block-RAM output, so mem[pos] is already latched when position pos is shifted in.
    // Registering pix_addr as well would cost two cycles of latency and shift every
    // window one position late -- which is exactly the bug the golden check caught, and
    // which a symmetric test image would have hidden.
    output wire [15:0]          pix_addr,
    input  wire [23:0]          pix_data,      // {B, G, R}
    output reg                  busy,
    output reg                  trigger,
    output reg  [15:0]          out_addr,
    output reg  signed [15:0]   out_data,
    output reg                  out_we,
    output wire                 leak_sink
);
    localparam integer OUT_SIDE  = IMG_SIDE - 2;
    localparam integer N_POS     = IMG_SIDE * IMG_SIDE;
    localparam integer N_OUTPUT  = OUT_SIDE * OUT_SIDE;
    localparam integer DWELL_W   = (DWELL <= 2) ? 1 : $clog2(DWELL);
    localparam integer SIDE_W    = $clog2(IMG_SIDE);

    localparam [1:0] MODE_SERIAL   = 2'd0;
    localparam [1:0] MODE_PARALLEL = 2'd1;
    localparam [1:0] MODE_SUMMED   = 2'd2;

    reg [SIDE_W:0] row, col;                 // one spare bit: row counts to IMG_SIDE
    reg [1:0] chan;
    reg [DWELL_W-1:0] dwell_count;
    reg [15:0] pos;                          // raster position being shifted in

    // ---------------------------------------------------------------- line buffer
    // Two delay lines of IMG_SIDE pixels. lb1 yields the pixel one row back, lb0 two rows
    // back. Inferred as shift registers (SRL) rather than RAM: no addressing, one in one
    // out, which is exactly what avoids the multiplexer blow-up of the previous design.
    reg [23:0] lb0 [0:IMG_SIDE-1];
    reg [23:0] lb1 [0:IMG_SIDE-1];
    integer li;

    wire [23:0] row1_pix = lb1[IMG_SIDE-1];  // (r-1, c)
    wire [23:0] row0_pix = lb0[IMG_SIDE-1];  // (r-2, c)

    // 3x3 window registers, win[r][c], row 0 oldest (top).
    reg [23:0] win [0:2][0:2];

    // The window is loaded at the START of a position's first dwell period, so for the
    // remainder of that position's periods the window holds the CURRENT position. Doing
    // it at the end instead left the window one position behind when the output was
    // emitted, which is an off-by-one that no functional check on a symmetric image
    // would catch.
    wire shift_now = busy && (dwell_count == {DWELL_W{1'b0}}) && (chan == 2'd0);

    always @(posedge clk) begin
        if (reset) begin
            for (li = 0; li < IMG_SIDE; li = li + 1) begin
                lb0[li] <= 24'd0;
                lb1[li] <= 24'd0;
            end
            win[0][0] <= 24'd0; win[0][1] <= 24'd0; win[0][2] <= 24'd0;
            win[1][0] <= 24'd0; win[1][1] <= 24'd0; win[1][2] <= 24'd0;
            win[2][0] <= 24'd0; win[2][1] <= 24'd0; win[2][2] <= 24'd0;
        end else if (shift_now) begin
            // Shift the window left and bring in the new column from the three taps.
            win[0][0] <= win[0][1]; win[0][1] <= win[0][2]; win[0][2] <= row0_pix;
            win[1][0] <= win[1][1]; win[1][1] <= win[1][2]; win[1][2] <= row1_pix;
            win[2][0] <= win[2][1]; win[2][1] <= win[2][2]; win[2][2] <= pix_data;
            // Advance both delay lines.
            lb0[0] <= row1_pix;
            for (li = 1; li < IMG_SIDE; li = li + 1) lb0[li] <= lb0[li-1];
            lb1[0] <= pix_data;
            for (li = 1; li < IMG_SIDE; li = li + 1) lb1[li] <= lb1[li-1];
        end
    end

    // ---------------------------------------------------------------- MAC
    function automatic signed [15:0] tap(input k, input [7:0] pix);
        tap = k ? $signed({8'd0, pix}) : -$signed({8'd0, pix});
    endfunction

    // Channel c of window position (r, col) -> kernel index c*9 + r*3 + col.
    wire signed [15:0] prod [0:26];
    genvar gc, gr, gk;
    generate
        for (gc = 0; gc < 3; gc = gc + 1) begin : g_ch
            for (gr = 0; gr < 3; gr = gr + 1) begin : g_row
                for (gk = 0; gk < 3; gk = gk + 1) begin : g_col
                    assign prod[gc*9 + gr*3 + gk] =
                        tap(kernel_bits[gc*9 + gr*3 + gk], win[gr][gk][gc*8 +: 8]);
                end
            end
        end
    endgenerate

    wire signed [15:0] chan_sum [0:2];
    generate
        for (gc = 0; gc < 3; gc = gc + 1) begin : g_csum
            assign chan_sum[gc] = prod[gc*9+0] + prod[gc*9+1] + prod[gc*9+2]
                                + prod[gc*9+3] + prod[gc*9+4] + prod[gc*9+5]
                                + prod[gc*9+6] + prod[gc*9+7] + prod[gc*9+8];
        end
    endgenerate

    wire signed [15:0] full_sum = chan_sum[0] + chan_sum[1] + chan_sum[2];
    wire signed [15:0] active_chan_sum = chan_sum[chan];

    // 27 product registers, clocked only in parallel mode; held in summed mode so they
    // contribute no switching activity. Serial clocks only the active channel's nine.
    reg signed [15:0] prod_reg [0:26];
    integer pi;
    always @(posedge clk) begin
        if (reset) begin
            for (pi = 0; pi < 27; pi = pi + 1) prod_reg[pi] <= 16'sd0;
        end else if (busy && mac_strobe && mode == MODE_PARALLEL) begin
            for (pi = 0; pi < 27; pi = pi + 1) prod_reg[pi] <= prod[pi];
        end else if (busy && mac_strobe && mode == MODE_SERIAL) begin
            for (pi = 0; pi < 9; pi = pi + 1)
                prod_reg[pi] <= prod[{30'd0, chan} * 9 + pi[31:0]];
        end
    end

    reg signed [15:0] acc;
    always @(posedge clk) begin
        if (reset) begin
            acc <= 16'sd0;
        end else if (busy && mac_strobe) begin
            if (mode == MODE_SERIAL)
                acc <= (chan == 2'd0) ? active_chan_sum : (acc + active_chan_sum);
            else
                acc <= full_sum;
        end
    end

    // The product registers must have an observable sink or synthesis deletes them.
    // Out-of-context synthesis of an earlier revision returned 362 registers where ~733
    // were expected: prod_reg drove nothing, so Vivado removed all 432 flops -- which
    // would have made MODE_PARALLEL and MODE_SUMMED electrically identical and silently
    // destroyed the comparison this core exists to make. Folding the bank into the
    // leak_sink reduction keeps it alive without altering out_data.
    wire prod_xor = ^{prod_reg[ 0], prod_reg[ 1], prod_reg[ 2], prod_reg[ 3],
                      prod_reg[ 4], prod_reg[ 5], prod_reg[ 6], prod_reg[ 7],
                      prod_reg[ 8], prod_reg[ 9], prod_reg[10], prod_reg[11],
                      prod_reg[12], prod_reg[13], prod_reg[14], prod_reg[15],
                      prod_reg[16], prod_reg[17], prod_reg[18], prod_reg[19],
                      prod_reg[20], prod_reg[21], prod_reg[22], prod_reg[23],
                      prod_reg[24], prod_reg[25], prod_reg[26]};

    // One flop, never a combinational path to a pad: the grey design proved that a
    // combinational leak_sink yields a bitstream that programs, meets timing and reads
    // back all zeros. Still no amplifier -- one flop, not a bank.
    reg leak_reg;
    always @(posedge clk) leak_reg <= reset ? 1'b0 : (^acc) ^ prod_xor;
    assign leak_sink = leak_reg;

    // ---------------------------------------------------------------- control
    // The window whose bottom-right corner is the position just shifted in is valid once
    // two full rows and two columns have passed.
    // dwell_count==0 loads the window; it is stable from dwell_count==1 onward, so the
    // MAC and the output are strobed then. DWELL >= 2 guarantees that cycle exists.
    wire mac_strobe = (dwell_count == {{(DWELL_W-1){1'b0}}, 1'b1});

    wire win_valid = (row >= 2) && (col >= 2);
    wire [15:0] valid_addr = (row - 2) * OUT_SIDE[15:0] + (col - 2);

    wire last_chan = (mode != MODE_SERIAL) || (chan == 2'd2);

    // Fetch one position ahead while running; hold 0 before the scan starts so that
    // mem[0] is latched in time for the very first shift.
    assign pix_addr = busy ? (pos + 16'd1) : 16'd0;

    initial begin
        if (DWELL < 2 || (DWELL % 2) != 0) $error("DWELL must be even and >= 2");
        if (IMG_SIDE < 4)                  $error("IMG_SIDE must be >= 4");
    end

    always @(posedge clk) begin
        if (reset) begin
            busy        <= 1'b0;
            trigger     <= 1'b0;
            out_addr    <= 16'd0;
            out_data    <= 16'sd0;
            out_we      <= 1'b0;
            row         <= {(SIDE_W+1){1'b0}};
            col         <= {(SIDE_W+1){1'b0}};
            chan        <= 2'd0;
            dwell_count <= {DWELL_W{1'b0}};
            pos         <= 16'd0;
        end else begin
            out_we <= 1'b0;
            if (start && !busy) begin
                busy        <= 1'b1;
                row         <= {(SIDE_W+1){1'b0}};
                col         <= {(SIDE_W+1){1'b0}};
                chan        <= 2'd0;
                dwell_count <= {DWELL_W{1'b0}};
                pos         <= 16'd0;
                trigger     <= (seg_len == 16'd0) || (seg_start == 16'd0);
            end else if (busy) begin
                // Emit on the final channel's dwell, so all three modes agree.
                if (mac_strobe && last_chan && win_valid) begin
                    out_data <= full_sum;
                    out_addr <= valid_addr;
                    out_we   <= 1'b1;
                end

                if (dwell_count == DWELL-1) begin
                    dwell_count <= {DWELL_W{1'b0}};
                    if (!last_chan) begin
                        chan <= chan + 2'd1;
                    end else begin
                        chan <= 2'd0;
                        if (pos == N_POS-1) begin
                            busy    <= 1'b0;
                            trigger <= 1'b0;
                        end else begin
                            pos      <= pos + 16'd1;
                            if (col == IMG_SIDE-1) begin
                                col <= {(SIDE_W+1){1'b0}};
                                row <= row + {{SIDE_W{1'b0}}, 1'b1};
                            end else begin
                                col <= col + {{SIDE_W{1'b0}}, 1'b1};
                            end
                            // Frame the trigger over the requested segment.
                            trigger <= (seg_len == 16'd0) ||
                                       (((pos + 16'd1) >= seg_start) &&
                                        ((pos + 16'd1) < (seg_start + seg_len)));
                        end
                    end
                end else begin
                    dwell_count <= dwell_count + {{(DWELL_W-1){1'b0}}, 1'b1};
                end
            end
        end
    end
endmodule

`default_nettype wire
