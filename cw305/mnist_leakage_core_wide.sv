`timescale 1ns/1ps
`default_nettype none

// Control design for the input-quantisation experiment: the binary convolution
// engine with a WIDE leakage amplifier.
//
// Identical to mnist_leakage_core.sv in every functional respect -- one bit per
// pixel, the same 9-bit match term, the same signed output. The only change is
// that the amplifier bank holds AMP_COPIES copies of match_bits instead of one,
// so it toggles the same number of flops as the grey core (72 x LEAK_LANES).
// Replicating a term adds amplitude but no information, which is exactly the
// control needed to tell "the grey design leaks more because the input carries
// more" apart from "the grey design leaks more because its bank is bigger".
//
// Each of 26x26 valid MNIST windows is held for DWELL clocks.  On alternating
// clocks, LEAK_LANES retained register banks transition 0 <-> XNOR(window,kernel).
// Transition count therefore follows the binary-convolution match count while
// the real signed convolution result is also written out.  LEAK_LANES is odd so
// the reduction sink is data dependent; DONT_TOUCH keeps the replicated flops.
module mnist_leakage_core_wide #(
    parameter integer DWELL      = 8,
    parameter integer LEAK_LANES = 63,
    parameter integer AMP_COPIES = 8    // 8 x 9 = 72 bits, matching the grey core
)(
    input  wire                 clk,
    input  wire                 reset,
    input  wire                 start,
    input  wire [783:0]         image_bits,
    input  wire [8:0]           kernel_bits,
    output reg                  busy,
    output reg                  trigger,
    output reg  [9:0]           out_addr,
    output reg  signed [7:0]    out_data,
    output reg                  out_we,
    output wire                 leak_sink
);
    localparam integer OUT_SIDE  = 26;
    localparam integer N_WINDOWS = OUT_SIDE * OUT_SIDE;
    localparam integer LEAK_BITS = 9 * AMP_COPIES * LEAK_LANES;
    localparam integer DWELL_W   = (DWELL <= 2) ? 1 : $clog2(DWELL);

    reg [4:0] row;
    reg [4:0] col;
    reg [DWELL_W-1:0] dwell_count;

    wire [9:0] base = row * 10'd28 + col;
    wire [8:0] window_bits = {
        image_bits[base + 10'd58], image_bits[base + 10'd57], image_bits[base + 10'd56],
        image_bits[base + 10'd30], image_bits[base + 10'd29], image_bits[base + 10'd28],
        image_bits[base + 10'd2],  image_bits[base + 10'd1],  image_bits[base]
    };
    wire [8:0] match_bits = ~(window_bits ^ kernel_bits);

    function automatic [3:0] popcount9(input [8:0] value);
        integer i;
        begin
            popcount9 = 4'd0;
            for (i = 0; i < 9; i = i + 1)
                popcount9 = popcount9 + value[i];
        end
    endfunction

    wire [3:0] match_count = popcount9(match_bits);
    wire signed [7:0] conv_value = $signed({3'b000, match_count, 1'b0}) - 8'sd9;

    (* DONT_TOUCH = "TRUE", KEEP = "TRUE" *) reg [LEAK_BITS-1:0] leak_bank;
    assign leak_sink = ^leak_bank;

    initial begin
        if (DWELL < 2 || (DWELL % 2) != 0)
            $error("DWELL must be even and >= 2");
        if ((LEAK_LANES % 2) == 0)
            $error("LEAK_LANES must be odd");
    end

    always @(posedge clk) begin
        if (reset) begin
            busy        <= 1'b0;
            trigger     <= 1'b0;
            out_addr    <= 10'd0;
            out_data    <= 8'sd0;
            out_we      <= 1'b0;
            row         <= 5'd0;
            col         <= 5'd0;
            dwell_count <= {DWELL_W{1'b0}};
            leak_bank   <= {LEAK_BITS{1'b0}};
        end else begin
            out_we <= 1'b0;
            if (start && !busy) begin
                busy        <= 1'b1;
                trigger     <= 1'b1;
                out_addr    <= 10'd0;
                row         <= 5'd0;
                col         <= 5'd0;
                dwell_count <= {DWELL_W{1'b0}};
                leak_bank   <= {LEAK_BITS{1'b0}};
            end else if (busy) begin
                // Every window begins from zero.  Odd/even dwell clocks alternate
                // between data and zero, producing repeated, aligned transitions.
                if (!dwell_count[0])
                    leak_bank <= {LEAK_LANES{{AMP_COPIES{match_bits}}}};
                else
                    leak_bank <= {LEAK_BITS{1'b0}};

                if (dwell_count == {DWELL_W{1'b0}}) begin
                    out_data <= conv_value;
                    out_we   <= 1'b1;
                end

                if (dwell_count == DWELL-1) begin
                    dwell_count <= {DWELL_W{1'b0}};
                    if (out_addr == N_WINDOWS-1) begin
                        busy      <= 1'b0;
                        trigger   <= 1'b0;
                        leak_bank <= {LEAK_BITS{1'b0}};
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
            end else begin
                leak_bank <= {LEAK_BITS{1'b0}};
            end
        end
    end
endmodule

`default_nettype wire
