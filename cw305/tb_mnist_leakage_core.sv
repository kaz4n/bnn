`timescale 1ns/1ps
`default_nettype none

module tb_mnist_leakage_core;
    localparam integer DWELL = 8;
    reg clk = 1'b0;
    reg reset = 1'b1;
    reg start = 1'b0;
    reg [783:0] image_bits = {784{1'b0}};
    reg [8:0] kernel_bits = 9'b0;
    wire busy, trigger, out_we, leak_sink;
    wire [9:0] out_addr;
    wire signed [7:0] out_data;
    integer writes = 0;
    integer busy_cycles = 0;

    always #5 clk = ~clk;

    mnist_leakage_core #(.DWELL(DWELL), .LEAK_LANES(3)) dut (
        .clk(clk), .reset(reset), .start(start), .image_bits(image_bits),
        .kernel_bits(kernel_bits), .busy(busy), .trigger(trigger),
        .out_addr(out_addr), .out_data(out_data), .out_we(out_we),
        .leak_sink(leak_sink)
    );

    always @(posedge clk) begin
        if (busy)
            busy_cycles = busy_cycles + 1;
        if (out_we) begin
            if (out_addr !== writes[9:0]) begin
                $display("FAIL address: got %0d expected %0d", out_addr, writes);
                $finish;
            end
            if (out_data !== 8'sd9) begin
                $display("FAIL data at %0d: got %0d expected 9", out_addr, out_data);
                $finish;
            end
            writes = writes + 1;
        end
    end

    initial begin
        repeat (4) @(posedge clk);
        reset <= 1'b0;
        @(posedge clk);
        start <= 1'b1;
        @(posedge clk);
        start <= 1'b0;
        wait (busy);
        wait (!busy);
        @(posedge clk);
        if (writes != 676) begin
            $display("FAIL writes: got %0d expected 676", writes);
            $finish;
        end
        if (busy_cycles != 676*DWELL) begin
            $display("FAIL busy cycles: got %0d expected %0d", busy_cycles, 676*DWELL);
            $finish;
        end
        if (trigger !== 1'b0) begin
            $display("FAIL trigger stuck high");
            $finish;
        end
        $display("PASS writes=%0d busy_cycles=%0d", writes, busy_cycles);
        $finish;
    end
endmodule

`default_nettype wire
