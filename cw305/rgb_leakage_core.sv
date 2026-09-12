`timescale 1ns/1ps
`default_nettype none

// RGB 3x3 convolution engine for CW305/CW-Lite side-channel research.
//
// Three-channel counterpart of mnist_leakage_core_grey_noamp.sv. The first layer
// multiplies real 0..255 RGB pixels by binary +/-1 weights, one weight per (channel,
// tap), exactly as a quantized first layer does. No leakage amplifier: this is what an
// attacker faces on an accelerator nobody modified, and it is the baseline any published
// number should be quoted against, since neither Wei et al. nor Power2Picture add such a
// circuit.
//
// RUNTIME-SELECTABLE CHANNEL DATAFLOW -- the point of this core
// -------------------------------------------------------------
// Phase 0 (experiments/rgb_sca/) measured that whether per-channel information survives
// the first-layer accumulation depends on what the hardware REGISTERS, not on the
// arithmetic. So the dataflow is a runtime mode, not a build-time parameter:
//
//   MODE_SERIAL   (0) One dwell period PER CHANNEL, 3 per output pixel. The accumulator
//                     runs across channels. Each channel gets its own settled dwell
//                     window, which is the design response to the inter-window smear
//                     measured on the reference capture -- adjacent channels are not
//                     forced into adjacent clock cycles.
//   MODE_PARALLEL (1) One dwell period. All 27 products are held in their own
//                     registers, so per-tap (hence per-channel) switching is observable.
//   MODE_SUMMED   (2) One dwell period. The product registers are NOT clocked, so only
//                     the accumulator toggles. Worst case for the attacker.
//
// Selecting at runtime rather than per-bitstream is deliberate: reprogramming the CW305
// over an already-configured FPGA silently fails on this board, which is exactly how the
// 2026-09-06 amplifier sweep produced four captures of the same bitstream
// (attack/hardtests_20260830/AMP_SWEEP_VOID.md). One bitstream, three modes, no
// reprogramming between conditions removes that failure mode entirely.
//
// The convolution RESULT is identical in all three modes. Only the registered state
// differs. A functional check that passes in one mode must pass in all three, and the
// host capture script asserts exactly that.
module rgb_leakage_core #(
    parameter integer DWELL    = 8,
    parameter integer IMG_SIDE = 32
)(
    input  wire                 clk,
    input  wire                 reset,
    input  wire                 start,
    input  wire [1:0]           mode,          // 0 serial, 1 parallel, 2 summed
    // The image lives in RAM in the register file, not in a flat vector here: holding
    // 3072 packed bytes in this module would cost a 1024-to-1 mux per tap and push the
    // USB write path past its timing constraint. The core publishes the window origin
    // and receives the 27 bytes back (same contract as the grey core, widened to RGB).
    output wire [9:0]           window_base,   // origin pixel index within one channel
    input  wire [215:0]         window_bytes,  // {ch2 p8..p0, ch1 p8..p0, ch0 p8..p0}
    input  wire [26:0]          kernel_bits,   // one +/-1 weight per (channel, tap)
    output reg                  busy,
    output reg                  trigger,
    output reg  [9:0]           out_addr,
    output reg  signed [15:0]   out_data,
    output reg                  out_we,
    output wire                 leak_sink
);
    localparam integer OUT_SIDE  = IMG_SIDE - 2;          // 'valid' 3x3
    localparam integer N_WINDOWS = OUT_SIDE * OUT_SIDE;
    localparam integer DWELL_W   = (DWELL <= 2) ? 1 : $clog2(DWELL);
    localparam integer ADDR_W    = 10;

    localparam [1:0] MODE_SERIAL   = 2'd0;
    localparam [1:0] MODE_PARALLEL = 2'd1;
    localparam [1:0] MODE_SUMMED   = 2'd2;

    reg [4:0] row;
    reg [4:0] col;
    reg [1:0] chan;                                        // serial mode only
    reg [DWELL_W-1:0] dwell_count;

    assign window_base = row * IMG_SIDE[9:0] + col;

    // ---------------------------------------------------------------- taps
    // Signed product of one tap: weight 1 adds the pixel, weight 0 subtracts it.
    function automatic signed [15:0] tap(input k, input [7:0] pix);
        tap = k ? $signed({8'd0, pix}) : -$signed({8'd0, pix});
    endfunction

    wire signed [15:0] prod [0:26];
    genvar gi;
    generate
        for (gi = 0; gi < 27; gi = gi + 1) begin : g_prod
            assign prod[gi] = tap(kernel_bits[gi], window_bytes[gi*8 +: 8]);
        end
    endgenerate

    // Per-channel partial sums. Range is +/- 9*255 = +/- 2295 per channel and
    // +/- 6885 over three, both inside 16 bits signed.
    wire signed [15:0] chan_sum [0:2];
    generate
        for (gi = 0; gi < 3; gi = gi + 1) begin : g_chan
            assign chan_sum[gi] = prod[gi*9+0] + prod[gi*9+1] + prod[gi*9+2]
                                + prod[gi*9+3] + prod[gi*9+4] + prod[gi*9+5]
                                + prod[gi*9+6] + prod[gi*9+7] + prod[gi*9+8];
        end
    endgenerate

    wire signed [15:0] full_sum = chan_sum[0] + chan_sum[1] + chan_sum[2];

    // The channel actually presented to the datapath this cycle. In serial mode only
    // the current channel's products are live; otherwise all three are.
    wire signed [15:0] active_chan_sum = chan_sum[chan];

    // ---------------------------------------------------------------- product bank
    // 27 product registers. Clocked ONLY in MODE_PARALLEL, so in MODE_SUMMED they hold
    // their value and contribute no switching activity. This is what makes the three
    // modes physically different rather than merely nominally different: the same
    // arithmetic, different registered state, different dynamic power.
    reg signed [15:0] prod_reg [0:26];
    integer pi;
    always @(posedge clk) begin
        if (reset) begin
            for (pi = 0; pi < 27; pi = pi + 1) prod_reg[pi] <= 16'sd0;
        end else if (busy && mode == MODE_PARALLEL) begin
            for (pi = 0; pi < 27; pi = pi + 1) prod_reg[pi] <= prod[pi];
        end else if (busy && mode == MODE_SERIAL) begin
            // Serial registers only the active channel's nine products, so the other
            // eighteen stay quiet -- that quietness is the per-channel signature.
            for (pi = 0; pi < 9; pi = pi + 1)
                prod_reg[pi] <= prod[{30'd0, chan} * 9 + pi[31:0]];
        end
    end

    // ---------------------------------------------------------------- accumulator
    // The registered accumulator, present in every mode. In serial mode it accumulates
    // across the three channel dwell periods; otherwise it takes the full sum at once.
    reg signed [15:0] acc;
    always @(posedge clk) begin
        if (reset) begin
            acc <= 16'sd0;
        end else if (busy) begin
            if (mode == MODE_SERIAL) begin
                if (dwell_count == {DWELL_W{1'b0}})
                    acc <= (chan == 2'd0) ? active_chan_sum : (acc + active_chan_sum);
            end else if (dwell_count == {DWELL_W{1'b0}}) begin
                acc <= full_sum;
            end
        end
    end

    // One flop, not a bank -- still no amplifier. Driving leak_sink combinationally from
    // the adder tree produced a bitstream that programmed cleanly, reported DONE and met
    // timing, yet read back all zeros from every register. See the grey core's note and
    // the leak-sink-must-be-registered finding.
    reg leak_reg;
    always @(posedge clk) leak_reg <= reset ? 1'b0 : ^acc;
    assign leak_sink = leak_reg;

    initial begin
        if (DWELL < 2 || (DWELL % 2) != 0)
            $error("DWELL must be even and >= 2");
        if (IMG_SIDE < 4)
            $error("IMG_SIDE must be >= 4");
        if (N_WINDOWS > (1 << ADDR_W))
            $error("N_WINDOWS exceeds out_addr width; widen ADDR_W");
    end

    wire last_chan = (mode != MODE_SERIAL) || (chan == 2'd2);

    always @(posedge clk) begin
        if (reset) begin
            busy        <= 1'b0;
            trigger     <= 1'b0;
            out_addr    <= {ADDR_W{1'b0}};
            out_data    <= 16'sd0;
            out_we      <= 1'b0;
            row         <= 5'd0;
            col         <= 5'd0;
            chan        <= 2'd0;
            dwell_count <= {DWELL_W{1'b0}};
        end else begin
            out_we <= 1'b0;
            if (start && !busy) begin
                busy        <= 1'b1;
                trigger     <= 1'b1;
                out_addr    <= {ADDR_W{1'b0}};
                row         <= 5'd0;
                col         <= 5'd0;
                chan        <= 2'd0;
                dwell_count <= {DWELL_W{1'b0}};
            end else if (busy) begin
                // The output is written once per window, on the final channel's dwell,
                // so the result is identical in all three modes.
                if (dwell_count == {DWELL_W{1'b0}} && last_chan) begin
                    out_data <= full_sum;
                    out_we   <= 1'b1;
                end

                if (dwell_count == DWELL-1) begin
                    dwell_count <= {DWELL_W{1'b0}};
                    if (!last_chan) begin
                        chan <= chan + 2'd1;      // serial: next channel, same window
                    end else begin
                        chan <= 2'd0;
                        if (out_addr == N_WINDOWS-1) begin
                            busy    <= 1'b0;
                            trigger <= 1'b0;
                        end else begin
                            out_addr <= out_addr + {{(ADDR_W-1){1'b0}}, 1'b1};
                            if (col == OUT_SIDE-1) begin
                                col <= 5'd0;
                                row <= row + 5'd1;
                            end else begin
                                col <= col + 5'd1;
                            end
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
