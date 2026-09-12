`timescale 1ns / 1ps
`default_nettype none
// RTL simulation only. A functional wire models this dedicated clock buffer;
// no device timing, placement, routing, or physical hardware is represented.
module BUFG(input wire I, output wire O);
    assign O = I;
endmodule
`default_nettype wire
