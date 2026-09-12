`timescale 1ns/1ps
`default_nettype none

// Testbench for rgb_leakage_core.
//
// The load-bearing property is the MODE INVARIANT: serial, parallel and summed must
// produce bit-identical convolution outputs while differing only in what they register.
// If that ever fails, a cross-mode comparison on hardware would be measuring an
// arithmetic difference rather than a dataflow difference, and every conclusion drawn
// from it would be wrong.
//
// Also checks the serial timing contract (3 dwell periods per window, not 1), because
// the host capture script slices traces by that geometry.
module tb_rgb_leakage_core;
    localparam integer DWELL    = 4;
    localparam integer IMG_SIDE = 8;                 // small, for a fast exhaustive check
    localparam integer OUT_SIDE = IMG_SIDE - 2;
    localparam integer N_WIN    = OUT_SIDE * OUT_SIDE;

    reg clk = 1'b0, reset = 1'b1, start = 1'b0;
    reg [1:0] mode = 2'd0;
    wire [9:0] window_base;
    reg [215:0] window_bytes;
    reg [26:0] kernel_bits;
    wire busy, trigger, out_we, leak_sink;
    wire [9:0] out_addr;
    wire signed [15:0] out_data;

    always #5 clk = ~clk;

    rgb_leakage_core #(.DWELL(DWELL), .IMG_SIDE(IMG_SIDE)) dut (
        .clk(clk), .reset(reset), .start(start), .mode(mode),
        .window_base(window_base), .window_bytes(window_bytes),
        .kernel_bits(kernel_bits), .busy(busy), .trigger(trigger),
        .out_addr(out_addr), .out_data(out_data), .out_we(out_we),
        .leak_sink(leak_sink)
    );

    // Image memory: 3 channels x IMG_SIDE x IMG_SIDE bytes.
    reg [7:0] img [0:3*IMG_SIDE*IMG_SIDE-1];
    integer i, c, t, errors, n_seen;
    integer cycles;
    reg signed [15:0] captured [0:N_WIN-1];
    reg signed [15:0] ref_serial [0:N_WIN-1];
    integer cycles_mode [0:2];

    // Combinationally serve the 27-byte window for the origin the core publishes.
    integer wr, wc, bi, br, bc;
    always @(*) begin
        wr = window_base / IMG_SIDE;
        wc = window_base % IMG_SIDE;
        for (c = 0; c < 3; c = c + 1)
            for (t = 0; t < 9; t = t + 1) begin
                br = wr + (t / 3);
                bc = wc + (t % 3);
                bi = c*IMG_SIDE*IMG_SIDE + br*IMG_SIDE + bc;
                window_bytes[(c*9+t)*8 +: 8] = img[bi];
            end
    end

    // Golden model, independent of the DUT's structure.
    function automatic signed [15:0] golden(input integer win);
        integer gr, gc, gcc, gt, grr, gccc, gidx;
        reg signed [15:0] s;
        begin
            gr = win / OUT_SIDE;
            gc = win % OUT_SIDE;
            s = 16'sd0;
            for (gcc = 0; gcc < 3; gcc = gcc + 1)
                for (gt = 0; gt < 9; gt = gt + 1) begin
                    grr  = gr + (gt / 3);
                    gccc = gc + (gt % 3);
                    gidx = gcc*IMG_SIDE*IMG_SIDE + grr*IMG_SIDE + gccc;
                    s = kernel_bits[gcc*9+gt] ? (s + $signed({8'd0, img[gidx]}))
                                              : (s - $signed({8'd0, img[gidx]}));
                end
            golden = s;
        end
    endfunction

    task run_mode(input [1:0] m);
        begin
            mode = m;
            n_seen = 0;
            cycles = 0;
            @(negedge clk); start = 1'b1;
            @(negedge clk); start = 1'b0;
            while (busy || n_seen < N_WIN) begin
                @(posedge clk);
                cycles = cycles + 1;
                if (out_we) begin
                    if (n_seen < N_WIN) captured[n_seen] = out_data;
                    n_seen = n_seen + 1;
                end
                if (cycles > 200000) begin
                    $display("TIMEOUT in mode %0d", m);
                    errors = errors + 1;
                    disable run_mode;
                end
            end
            cycles_mode[m] = cycles;
            $display("  mode %0d: %0d outputs in %0d cycles", m, n_seen, cycles);
            if (n_seen != N_WIN) begin
                $display("  FAIL mode %0d: expected %0d outputs, got %0d", m, N_WIN, n_seen);
                errors = errors + 1;
            end
        end
    endtask

    task check_against_golden(input [1:0] m);
        integer w;
        begin
            for (w = 0; w < N_WIN; w = w + 1)
                if (captured[w] !== golden(w)) begin
                    $display("  FAIL mode %0d win %0d: got %0d expected %0d",
                             m, w, captured[w], golden(w));
                    errors = errors + 1;
                end
        end
    endtask

    integer seed;
    integer w;
    initial begin
        errors = 0;
        seed = 32'h1234_5678;

        for (i = 0; i < 3*IMG_SIDE*IMG_SIDE; i = i + 1)
            img[i] = $random(seed) & 8'hFF;
        kernel_bits = $random(seed);

        repeat (4) @(negedge clk);
        reset = 1'b0;
        repeat (2) @(negedge clk);

        $display("RGB leakage core: mode invariant check (IMG_SIDE=%0d, DWELL=%0d)",
                 IMG_SIDE, DWELL);

        run_mode(2'd0);
        check_against_golden(2'd0);
        for (w = 0; w < N_WIN; w = w + 1) ref_serial[w] = captured[w];

        run_mode(2'd1);
        check_against_golden(2'd1);
        for (w = 0; w < N_WIN; w = w + 1)
            if (captured[w] !== ref_serial[w]) begin
                $display("  FAIL: parallel differs from serial at win %0d", w);
                errors = errors + 1;
            end

        run_mode(2'd2);
        check_against_golden(2'd2);
        for (w = 0; w < N_WIN; w = w + 1)
            if (captured[w] !== ref_serial[w]) begin
                $display("  FAIL: summed differs from serial at win %0d", w);
                errors = errors + 1;
            end

        // Serial spends one dwell period per channel, so it must take about three times
        // as long. The host slices traces by this geometry, so it is asserted here.
        $display("  cycles serial=%0d parallel=%0d summed=%0d",
                 cycles_mode[0], cycles_mode[1], cycles_mode[2]);
        if (cycles_mode[1] != cycles_mode[2]) begin
            $display("  FAIL: parallel and summed must take the same number of cycles");
            errors = errors + 1;
        end
        if (cycles_mode[0] < 2*cycles_mode[1]) begin
            $display("  FAIL: serial must take ~3x the cycles of the single-pass modes");
            errors = errors + 1;
        end

        if (errors == 0)
            $display("PASS: all three modes bit-identical and matching the golden model");
        else
            $display("FAIL: %0d error(s)", errors);
        $finish;
    end
endmodule

`default_nettype wire
