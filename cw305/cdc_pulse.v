// Minimal pulse CDC (src_clk pulse -> one dst_clk pulse). Matches the cw305_reg_aes
// U_go_pulse interface. Toggle-synchronizer style.
`default_nettype none
`timescale 1ns/1ps
module cdc_pulse (
    input  wire reset_i,
    input  wire src_clk,
    input  wire src_pulse,
    input  wire dst_clk,
    output wire dst_pulse
);
    reg src_tog;
    always @(posedge src_clk or posedge reset_i)
        if (reset_i) src_tog <= 1'b0;
        else if (src_pulse) src_tog <= ~src_tog;

    (* ASYNC_REG="TRUE" *) reg [2:0] sync;
    always @(posedge dst_clk or posedge reset_i)
        if (reset_i) sync <= 3'b0;
        else sync <= {sync[1:0], src_tog};

    assign dst_pulse = sync[2] ^ sync[1];
endmodule
`default_nettype wire
