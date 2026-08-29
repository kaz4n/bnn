//============================================================================
// Ring-oscillator counter sensor for on-chip voltage/power observation.
//
// Each RO increments an asynchronous counter. The crypto clock samples Gray-coded
// counters, converts them back to binary, and reports the summed per-cycle delta.
// This produces one scalar leakage feature per sampled BNN output cycle.
//============================================================================
`timescale 1ns/1ps
`default_nettype none

(* KEEP_HIERARCHY = "yes", DONT_TOUCH = "true" *)
module ro_async_counter #(
    parameter integer STAGES = 5,
    parameter integer WIDTH  = 16
)(
    input  wire             enable,
    output wire             tap,
    output wire [WIDTH-1:0] gray_count
);
    // Build the ring from explicit LUT primitives so Vivado actually realises an
    // oscillating combinational loop (RTL ~assign chains get optimised / tied off).
    (* KEEP = "true", DONT_TOUCH = "true" *) wire [STAGES-1:0] ro_loop;

    // stage 0: O = enable & ~feedback  (INIT for I1=enable,I0=fb -> O=1 at 10 -> 4'h4)
    (* DONT_TOUCH = "true" *)
    LUT2 #(.INIT(4'h4)) U_inv0 (
        .O (ro_loop[0]),
        .I0(ro_loop[STAGES-1]),
        .I1(enable)
    );
    genvar s;
    generate
        for (s = 1; s < STAGES; s = s + 1) begin : GEN_INV
            (* DONT_TOUCH = "true" *)
            LUT1 #(.INIT(2'b01)) U_inv (        // O = ~I0
                .O (ro_loop[s]),
                .I0(ro_loop[s-1])
            );
        end
    endgenerate

    assign tap = ro_loop[STAGES-1];
    assign gray_count = {WIDTH{tap}};
endmodule


(* KEEP_HIERARCHY = "yes", DONT_TOUCH = "true" *)
module ro_counter_sensor #(
    parameter integer N_RO         = 16,
    parameter integer COUNTER_BITS = 16,
    parameter integer SAMPLE_BITS  = 16
)(
    input  wire                   clk,
    input  wire                   reset,
    input  wire                   enable,
    input  wire                   sample_en,
    output wire [SAMPLE_BITS-1:0] sample_value
);
    wire [N_RO-1:0] ro_tap_async;

    genvar r;
    generate
        for (r = 0; r < N_RO; r = r + 1) begin : GEN_RO
            localparam integer THIS_STAGES = 5 + 2*(r % 4);
            ro_async_counter #(.STAGES(THIS_STAGES), .WIDTH(COUNTER_BITS)) U_RO (
                .enable(enable),
                .tap(ro_tap_async[r]),
                .gray_count()
            );
        end
    endgenerate

    (* ASYNC_REG = "TRUE" *) reg [N_RO-1:0] tap_meta = {N_RO{1'b0}};
    (* ASYNC_REG = "TRUE" *) reg [N_RO-1:0] tap_sync = {N_RO{1'b0}};
    reg [N_RO-1:0] tap_prev = {N_RO{1'b0}};

    integer i;
    reg [31:0] sum_delta;
    always @* begin
        sum_delta = 32'd0;
        for (i = 0; i < N_RO; i = i + 1)
            sum_delta = sum_delta + (tap_sync[i] ^ tap_prev[i]);
    end

    assign sample_value = sum_delta[SAMPLE_BITS-1:0];

    integer j;
    always @(posedge clk) begin
        tap_meta <= ro_tap_async;
        tap_sync <= tap_meta;
        if (reset) begin
            tap_prev <= {N_RO{1'b0}};
        end else if (enable && sample_en) begin
            tap_prev <= tap_sync;
        end
    end
endmodule

`default_nettype wire
