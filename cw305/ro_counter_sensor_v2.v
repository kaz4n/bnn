//============================================================================
// ro_counter_sensor v2 -- REAL oscillation-counting voltage/droop sensor.
//
// DROP-IN replacement for cw305/ro_counter_sensor.v (same module name + ports).
// To use: in the build, add THIS file instead of ro_counter_sensor.v, rebuild.
//
// WHY: the v1 sensor sampled only the instantaneous RO tap PHASE and reported
//   sum_i (tap[i] ^ tap_prev[i]) -> a 0..N_RO phase-flip count (~4 bits, coarse).
//   The de-risk (RO_CHANNEL_RESULTS.md) showed that coarse encoding is the
//   SECOND factor dragging S6 background detection below its dilution-limit
//   (hw 0.255 vs sim 0.45). v2 counts each RO's OSCILLATIONS over the sample
//   window (count ∝ RO frequency ∝ 1/Vdd) -> ~10x finer droop reading, which
//   should recover S6. (It does NOT beat the S7/corr dilution wall -- that needs
//   the ro_pblock.tcl co-location + clock-gating.)
//
// !!! NEEDS VALIDATION: this clocks small fabric counters from unconstrained RO
//   nets (a generated/gated clock). It is the standard on-chip-sensor pattern
//   but MUST be checked: (1) C/behavioral sanity, (2) that Vivado keeps the ROs
//   and counters (DONT_TOUCH set), (3) on-silicon that sample_value now spans a
//   wide range (tens-hundreds), not 0..16. Keep ALLOW_COMBINATORIAL_LOOPS
//   (cw305/ro_sensor.xdc already globs *ro_loop*). Gray-coding + double-flop
//   makes the async read metastability-safe (<=1 bit changes per RO tick).
//============================================================================
`timescale 1ns/1ps
`default_nettype none

(* KEEP_HIERARCHY = "yes", DONT_TOUCH = "true" *)
module ro_async_counter #(
    parameter integer STAGES = 5,
    parameter integer WIDTH  = 12
)(
    input  wire             enable,
    output wire             tap,
    output wire [WIDTH-1:0] gray_count      // Gray-coded oscillation count (async domain)
);
    (* KEEP = "true", DONT_TOUCH = "true" *) wire [STAGES-1:0] ro_loop;

    // stage 0: O = enable & ~feedback  (LUT2 INIT 4'h4)
    (* DONT_TOUCH = "true" *)
    LUT2 #(.INIT(4'h4)) U_inv0 (.O(ro_loop[0]), .I0(ro_loop[STAGES-1]), .I1(enable));
    genvar s;
    generate
        for (s = 1; s < STAGES; s = s + 1) begin : GEN_INV
            (* DONT_TOUCH = "true" *)
            LUT1 #(.INIT(2'b01)) U_inv (.O(ro_loop[s]), .I0(ro_loop[s-1]));   // O = ~I0
        end
    endgenerate
    assign tap = ro_loop[STAGES-1];

    // Binary counter clocked by the RO tap (this is the oscillation count). The
    // tap is an unconstrained generated clock; the counter is tiny + local.
    (* DONT_TOUCH = "true" *) reg [WIDTH-1:0] bin = {WIDTH{1'b0}};
    always @(posedge tap or negedge enable) begin
        if (!enable) bin <= {WIDTH{1'b0}};
        else         bin <= bin + 1'b1;
    end
    // binary -> Gray so a mid-flight async sample differs from a settled value in
    // at most one bit (double-flop below then yields a valid adjacent count).
    assign gray_count = bin ^ (bin >> 1);
endmodule


(* KEEP_HIERARCHY = "yes", DONT_TOUCH = "true" *)
module ro_counter_sensor #(
    parameter integer N_RO         = 16,
    parameter integer COUNTER_BITS = 12,
    parameter integer SAMPLE_BITS  = 16
)(
    input  wire                   clk,          // crypto_clk (sample domain)
    input  wire                   reset,
    input  wire                   enable,       // ROs run while high (= busy)
    input  wire                   sample_en,    // 1 pulse per BNN output cycle
    output wire [SAMPLE_BITS-1:0] sample_value  // sum of per-RO oscillations this window
);
    // --- N_RO ring oscillators, each with its own Gray oscillation counter ---
    wire [COUNTER_BITS-1:0] gray_cnt [0:N_RO-1];
    genvar r;
    generate
        for (r = 0; r < N_RO; r = r + 1) begin : GEN_RO
            localparam integer THIS_STAGES = 5 + 2*(r % 4);   // varied lengths
            ro_async_counter #(.STAGES(THIS_STAGES), .WIDTH(COUNTER_BITS)) U_RO (
                .enable(enable), .tap(), .gray_count(gray_cnt[r])
            );
        end
    endgenerate

    function [COUNTER_BITS-1:0] gray2bin(input [COUNTER_BITS-1:0] g);
        integer k; reg [COUNTER_BITS-1:0] b;
        begin
            b = g;
            for (k = 1; k < COUNTER_BITS; k = k + 1) b = b ^ (g >> k);
            gray2bin = b;
        end
    endfunction

    integer i;
    reg [COUNTER_BITS-1:0] g_meta [0:N_RO-1];
    reg [COUNTER_BITS-1:0] g_sync [0:N_RO-1];
    reg [COUNTER_BITS-1:0] bin_prev [0:N_RO-1];
    reg [SAMPLE_BITS-1:0]  sum_reg = {SAMPLE_BITS{1'b0}};
    assign sample_value = sum_reg;

    // local, non-registered accumulator (blocking) -> registered once at clock edge
    reg [SAMPLE_BITS-1:0]  s_acc;
    reg [COUNTER_BITS-1:0] b_now;

    always @(posedge clk) begin
        // double-flop each Gray counter into the sample clock domain (metastable-safe)
        for (i = 0; i < N_RO; i = i + 1) begin
            g_meta[i] <= gray_cnt[i];
            g_sync[i] <= g_meta[i];
        end
        if (reset) begin
            for (i = 0; i < N_RO; i = i + 1) bin_prev[i] <= {COUNTER_BITS{1'b0}};
            sum_reg <= {SAMPLE_BITS{1'b0}};
        end else if (enable && sample_en) begin
            // per-RO oscillations since last sample = modular diff (handles wrap)
            s_acc = {SAMPLE_BITS{1'b0}};                       // blocking: local temp
            for (i = 0; i < N_RO; i = i + 1) begin
                b_now = gray2bin(g_sync[i]);
                s_acc = s_acc + ((b_now - bin_prev[i]) & {COUNTER_BITS{1'b1}});
                bin_prev[i] <= b_now;                          // non-blocking: state
            end
            sum_reg <= s_acc;                                  // non-blocking: output
        end
    end
endmodule

`default_nettype wire
