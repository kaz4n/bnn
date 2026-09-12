`timescale 1ns/1ps
`default_nettype none

// Self-checking testbench for the grey-pixel core.
//
// The hardware is not available while this is written, so the datapath is
// verified in simulation instead: a pseudo-random 0..255 image and a non-trivial
// kernel are driven in, and every one of the 676 outputs is compared against an
// independently computed reference (kernel bit 1 adds the pixel, 0 subtracts).
// It also checks the sequencing that the capture relies on: 676 writes in
// address order, DWELL cycles per window, and trigger low at the end.
module tb_mnist_leakage_core_grey;
    localparam integer DWELL    = 8;
    localparam integer OUT_SIDE = 26;
    localparam integer LINE     = 28;

    reg clk = 1'b0;
    reg reset = 1'b1;
    reg start = 1'b0;
    reg [8:0] kernel_bits = 9'b101101001;

    wire busy, trigger, out_we, leak_sink;
    wire [9:0] out_addr;
    wire signed [15:0] out_data;
    wire [9:0] win_base;

    integer writes = 0;
    integer busy_cycles = 0;
    integer errors = 0;

    reg [7:0] pix [0:LINE*LINE-1];

    always #5 clk = ~clk;

    // Mirrors what the register file does: nine RAM reads at the published origin.
    wire [71:0] win_bytes = {
        pix[win_base + 58], pix[win_base + 57], pix[win_base + 56],
        pix[win_base + 30], pix[win_base + 29], pix[win_base + 28],
        pix[win_base +  2], pix[win_base +  1], pix[win_base +  0]
    };

    mnist_leakage_core_grey #(.DWELL(DWELL), .LEAK_LANES(3)) dut (
        .clk(clk), .reset(reset), .start(start),
        .window_base(win_base), .window_bytes(win_bytes),
        .kernel_bits(kernel_bits), .busy(busy), .trigger(trigger),
        .out_addr(out_addr), .out_data(out_data), .out_we(out_we),
        .leak_sink(leak_sink)
    );

    // Independent reference for one output index.
    function automatic signed [31:0] expected(input integer idx);
        integer y, x, ky, kx, tap;
        integer acc;
        begin
            y = idx / OUT_SIDE;
            x = idx % OUT_SIDE;
            acc = 0;
            tap = 0;
            for (ky = 0; ky < 3; ky = ky + 1)
                for (kx = 0; kx < 3; kx = kx + 1) begin
                    tap = ky * 3 + kx;
                    if (kernel_bits[tap])
                        acc = acc + pix[(y + ky) * LINE + (x + kx)];
                    else
                        acc = acc - pix[(y + ky) * LINE + (x + kx)];
                end
            expected = acc;
        end
    endfunction

    reg signed [31:0] exp_full;
    reg signed [15:0] exp_word;
    always @(posedge clk) begin
        if (busy)
            busy_cycles = busy_cycles + 1;
        if (out_we) begin
            if (out_addr !== writes[9:0]) begin
                $display("FAIL address: got %0d expected %0d", out_addr, writes);
                errors = errors + 1;
            end
            exp_full = expected(writes);
            exp_word = exp_full[15:0];
            if (out_data !== exp_word) begin
                $display("FAIL data at %0d: got %0d expected %0d",
                         out_addr, out_data, exp_full);
                errors = errors + 1;
                if (errors > 8) $finish;
            end
            writes = writes + 1;
        end
    end

    integer i;
    integer seed = 32'd12345;
    initial begin
        // Pseudo-random grey image; win_bytes reads straight out of pix[].
        for (i = 0; i < LINE*LINE; i = i + 1)
            pix[i] = $random(seed) & 8'hFF;

        repeat (4) @(posedge clk);
        reset <= 1'b0;
        @(posedge clk);
        start <= 1'b1;
        @(posedge clk);
        start <= 1'b0;
        wait (busy);
        wait (!busy);
        @(posedge clk);

        if (writes != OUT_SIDE*OUT_SIDE) begin
            $display("FAIL writes: got %0d expected %0d", writes, OUT_SIDE*OUT_SIDE);
            errors = errors + 1;
        end
        if (busy_cycles != OUT_SIDE*OUT_SIDE*DWELL) begin
            $display("FAIL busy cycles: got %0d expected %0d",
                     busy_cycles, OUT_SIDE*OUT_SIDE*DWELL);
            errors = errors + 1;
        end
        if (trigger !== 1'b0) begin
            $display("FAIL trigger did not clear");
            errors = errors + 1;
        end

        if (errors == 0)
            $display("PASS grey core: %0d outputs bit-exact, %0d busy cycles",
                     writes, busy_cycles);
        else
            $display("FAILED with %0d errors", errors);
        $finish;
    end
endmodule

`default_nettype wire
