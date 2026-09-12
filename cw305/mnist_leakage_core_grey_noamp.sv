`timescale 1ns/1ps
`default_nettype none

// UNAMPLIFIED grey-pixel 3x3 convolution engine.
//
// Identical to mnist_leakage_core_grey.sv except the leakage amplifier is gone:
// no LEAK_LANES bank, nothing instantiated whose only purpose is to toggle on the
// MAC term. This is what an attacker faces on an accelerator nobody modified, and
// it is the baseline the published numbers should be quoted against, since neither
// Wei et al. nor Power2Picture add such a circuit.
//
// Separate file with the same module name, selected by
// run_vivado_leakage_grey_noamp.tcl, so the amplified core is never edited.
//
// Original header follows.
//
// Grey-pixel 3x3 convolution engine for CW305/CW-Lite research.
//
// This is the grey-input counterpart of mnist_leakage_core.sv.  The binary core
// takes one bit per pixel; this one takes a full 8-bit pixel, so the first layer
// multiplies real 0..255 pixels by binary +/-1 weights exactly as in Wei et al.
// The binary core is kept unchanged: every result in the 30 August report was
// produced with it, and the two designs are not interchangeable.
//
// Leakage amplifier.  The binary core toggles LEAK_LANES banks of
//     match_bits = ~(window_bits ^ kernel_bits)                        (9 bits)
// This core toggles the byte-wise generalisation of exactly that expression,
//     match_bytes = ~(window_bytes ^ {8{kernel_bit}} per tap)         (72 bits)
// For a 1-bit pixel the two are the same function, so the amplifier still rides
// a kernel-dependent MAC term rather than the raw pixel.  Riding the 16-bit sum
// instead would collapse the per-tap grey information the template attack needs.
module mnist_leakage_core_grey #(
    parameter integer DWELL      = 8,
    parameter integer LEAK_LANES = 63
)(
    input  wire                 clk,
    input  wire                 reset,
    input  wire                 start,
    // The image lives in a RAM array in the register file, not in a flat vector
    // here.  Holding 784 packed bytes in this module cost a 784-to-1 8-bit mux
    // per tap (26k LUTs) and pushed the USB write path past its 100 MHz
    // constraint.  The core now publishes the window origin and receives the
    // nine bytes back.
    output wire [9:0]           window_base,
    input  wire [71:0]          window_bytes, // {p8,...,p0}, p0 at the origin
    input  wire [8:0]           kernel_bits,
    output reg                  busy,
    output reg                  trigger,
    output reg  [9:0]           out_addr,
    output reg  signed [15:0]   out_data,
    output reg                  out_we,
    output wire                 leak_sink
);
    localparam integer OUT_SIDE  = 26;
    localparam integer N_WINDOWS = OUT_SIDE * OUT_SIDE;
    localparam integer LEAK_BITS = 72 * LEAK_LANES;
    localparam integer DWELL_W   = (DWELL <= 2) ? 1 : $clog2(DWELL);

    reg [4:0] row;
    reg [4:0] col;
    reg [DWELL_W-1:0] dwell_count;

    assign window_base = row * 10'd28 + col;

    // The nine taps of the valid 3x3 window, same offsets as the binary core.
    wire [7:0] p0 = window_bytes[  7:  0];
    wire [7:0] p1 = window_bytes[ 15:  8];
    wire [7:0] p2 = window_bytes[ 23: 16];
    wire [7:0] p3 = window_bytes[ 31: 24];
    wire [7:0] p4 = window_bytes[ 39: 32];
    wire [7:0] p5 = window_bytes[ 47: 40];
    wire [7:0] p6 = window_bytes[ 55: 48];
    wire [7:0] p7 = window_bytes[ 63: 56];
    wire [7:0] p8 = window_bytes[ 71: 64];

    wire [71:0] kernel_mask  = {{8{kernel_bits[8]}}, {8{kernel_bits[7]}},
                                {8{kernel_bits[6]}}, {8{kernel_bits[5]}},
                                {8{kernel_bits[4]}}, {8{kernel_bits[3]}},
                                {8{kernel_bits[2]}}, {8{kernel_bits[1]}},
                                {8{kernel_bits[0]}}};
    wire [71:0] match_bytes = ~(window_bytes ^ kernel_mask);

    // Real signed convolution: kernel bit 1 adds the pixel, bit 0 subtracts it.
    // Range is +/- 9*255 = +/- 2295, which needs 13 bits; 16 keeps the host-side
    // readback a clean two bytes per output.
    function automatic signed [15:0] tap(input k, input [7:0] pix);
        tap = k ? $signed({8'd0, pix}) : -$signed({8'd0, pix});
    endfunction

    wire signed [15:0] conv_value =
          tap(kernel_bits[0], p0) + tap(kernel_bits[1], p1) + tap(kernel_bits[2], p2)
        + tap(kernel_bits[3], p3) + tap(kernel_bits[4], p4) + tap(kernel_bits[5], p5)
        + tap(kernel_bits[6], p6) + tap(kernel_bits[7], p7) + tap(kernel_bits[8], p8);

    // One flop, not a bank. Driving leak_sink combinationally from conv_value put a
    // path from distributed RAM through nine 16-bit adders straight to a top-level pad;
    // every build that works drives that pin from a register, and the combinational
    // version produced a bitstream whose register interface was dead on hardware even
    // though this core passes its testbench. One flop is still no amplifier: the
    // smallest working amplified build has 72.
    reg leak_reg;
    always @(posedge clk) leak_reg <= reset ? 1'b0 : ^conv_value;
    assign leak_sink = leak_reg;

    initial begin
        if (DWELL < 2 || (DWELL % 2) != 0)
            $error("DWELL must be even and >= 2");
        // LEAK_LANES is unused in this variant; it stays on the parameter list only
        // so the register file and top level instantiate this core unchanged.
    end

    always @(posedge clk) begin
        if (reset) begin
            busy        <= 1'b0;
            trigger     <= 1'b0;
            out_addr    <= 10'd0;
            out_data    <= 16'sd0;
            out_we      <= 1'b0;
            row         <= 5'd0;
            col         <= 5'd0;
            dwell_count <= {DWELL_W{1'b0}};
        end else begin
            out_we <= 1'b0;
            if (start && !busy) begin
                busy        <= 1'b1;
                trigger     <= 1'b1;
                out_addr    <= 10'd0;
                row         <= 5'd0;
                col         <= 5'd0;
                dwell_count <= {DWELL_W{1'b0}};
            end else if (busy) begin
                // Every window begins from zero.  Odd/even dwell clocks alternate
                // between data and zero, producing repeated, aligned transitions.
                if (dwell_count == {DWELL_W{1'b0}}) begin
                    out_data <= conv_value;
                    out_we   <= 1'b1;
                end

                if (dwell_count == DWELL-1) begin
                    dwell_count <= {DWELL_W{1'b0}};
                    if (out_addr == N_WINDOWS-1) begin
                        busy      <= 1'b0;
                        trigger   <= 1'b0;
                    end else begin
                        out_addr <= out_addr + 10'd1;
                        if (col == OUT_SIDE-1) begin
                            col <= 5'd0;
                            row <= row + 5'd1;
                        end else begin
                            col <= col + 5'd1;
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
