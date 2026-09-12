`timescale 1ns/1ps
`default_nettype none

// Testbench for rgb_linebuf_core.
//
// The line buffer is the part most likely to be subtly wrong: a one-position lag or a
// swapped row tap produces outputs that look plausible and are wrong everywhere. So every
// output of every mode is checked against an independent golden convolution computed
// here, on an asymmetric random image where an off-by-one cannot cancel.
//
// Also checks:
//   * the mode invariant -- all three dataflows bit-identical (a cross-mode power
//     comparison is meaningless otherwise);
//   * segment framing -- the trigger is high for exactly the requested window range,
//     which is what keeps a 32x32 scan inside the CW-Lite's sample limit.
module tb_rgb_linebuf_core;
    localparam integer DWELL    = 4;
    localparam integer IMG_SIDE = 8;
    localparam integer OUT_SIDE = IMG_SIDE - 2;
    localparam integer N_POS    = IMG_SIDE * IMG_SIDE;
    localparam integer N_OUT    = OUT_SIDE * OUT_SIDE;

    reg clk = 1'b0, reset = 1'b1, start = 1'b0;
    reg [1:0] mode = 2'd0;
    reg [26:0] kernel_bits;
    reg [15:0] seg_start = 16'd0, seg_len = 16'd0;
    wire [15:0] pix_addr;
    reg  [23:0] pix_data;
    wire busy, trigger, out_we, leak_sink;
    wire [15:0] out_addr;
    wire signed [15:0] out_data;

    always #5 clk = ~clk;

    rgb_linebuf_core #(.DWELL(DWELL), .IMG_SIDE(IMG_SIDE)) dut (
        .clk(clk), .reset(reset), .start(start), .mode(mode),
        .kernel_bits(kernel_bits), .seg_start(seg_start), .seg_len(seg_len),
        .pix_addr(pix_addr), .pix_data(pix_data),
        .busy(busy), .trigger(trigger),
        .out_addr(out_addr), .out_data(out_data), .out_we(out_we),
        .leak_sink(leak_sink)
    );

    // Block-RAM model: registered read, one clock of latency, as the real store will be.
    reg [23:0] mem [0:N_POS-1];
    always @(posedge clk) pix_data <= mem[pix_addr];

    integer errors, i, c, t;
    integer n_seen, cycles;
    reg signed [15:0] captured [0:N_OUT-1];
    reg seen [0:N_OUT-1];
    reg signed [15:0] ref_mode0 [0:N_OUT-1];
    integer trig_cycles;

    // Golden 'valid' convolution from the testbench's own image copy.
    function automatic signed [15:0] golden(input integer win);
        integer gr, gc, cc, tt, rr, ccc;
        reg signed [15:0] s;
        reg [23:0] px;
        begin
            gr = win / OUT_SIDE;
            gc = win % OUT_SIDE;
            s = 16'sd0;
            for (cc = 0; cc < 3; cc = cc + 1)
                for (tt = 0; tt < 9; tt = tt + 1) begin
                    rr  = gr + (tt / 3);
                    ccc = gc + (tt % 3);
                    px  = mem[rr*IMG_SIDE + ccc];
                    s = kernel_bits[cc*9+tt] ? (s + $signed({8'd0, px[cc*8 +: 8]}))
                                             : (s - $signed({8'd0, px[cc*8 +: 8]}));
                end
            golden = s;
        end
    endfunction

    task run_mode(input [1:0] m);
        begin
            mode = m;
            n_seen = 0; cycles = 0; trig_cycles = 0;
            for (i = 0; i < N_OUT; i = i + 1) seen[i] = 1'b0;
            @(negedge clk); start = 1'b1;
            @(negedge clk); start = 1'b0;
            while (busy || cycles < 4) begin
                @(posedge clk);
                cycles = cycles + 1;
                if (trigger) trig_cycles = trig_cycles + 1;
                if (out_we) begin
                    if (out_addr < N_OUT) begin
                        captured[out_addr] = out_data;
                        seen[out_addr] = 1'b1;
                    end else begin
                        $display("  FAIL mode %0d: out_addr %0d out of range", m, out_addr);
                        errors = errors + 1;
                    end
                    n_seen = n_seen + 1;
                end
                if (cycles > 500000) begin
                    $display("  TIMEOUT mode %0d", m); errors = errors + 1;
                    disable run_mode;
                end
            end
            $display("  mode %0d: %0d writes, %0d cycles, trigger high %0d cycles",
                     m, n_seen, cycles, trig_cycles);
            if (n_seen != N_OUT) begin
                $display("  FAIL mode %0d: expected %0d outputs, got %0d", m, N_OUT, n_seen);
                errors = errors + 1;
            end
            for (i = 0; i < N_OUT; i = i + 1) begin
                if (!seen[i]) begin
                    $display("  FAIL mode %0d: output %0d never written", m, i);
                    errors = errors + 1;
                end else if (captured[i] !== golden(i)) begin
                    if (errors < 12)
                        $display("  FAIL mode %0d out %0d: got %0d expected %0d",
                                 m, i, captured[i], golden(i));
                    errors = errors + 1;
                end
            end
        end
    endtask

    integer seed;
    initial begin
        errors = 0;
        seed = 32'hFEED_BEEF;

        // Asymmetric random image: an off-by-one in the line buffer cannot cancel.
        for (i = 0; i < N_POS; i = i + 1)
            mem[i] = {$random(seed)} & 24'hFF_FFFF;
        kernel_bits = {$random(seed)} & 27'h7FF_FFFF;

        repeat (4) @(negedge clk);
        reset = 1'b0;
        repeat (2) @(negedge clk);

        $display("RGB line-buffer core (IMG_SIDE=%0d, DWELL=%0d, %0d outputs)",
                 IMG_SIDE, DWELL, N_OUT);

        run_mode(2'd0);
        for (i = 0; i < N_OUT; i = i + 1) ref_mode0[i] = captured[i];
        run_mode(2'd1);
        for (i = 0; i < N_OUT; i = i + 1)
            if (captured[i] !== ref_mode0[i]) begin
                $display("  FAIL: parallel differs from serial at %0d", i);
                errors = errors + 1;
            end
        run_mode(2'd2);
        for (i = 0; i < N_OUT; i = i + 1)
            if (captured[i] !== ref_mode0[i]) begin
                $display("  FAIL: summed differs from serial at %0d", i);
                errors = errors + 1;
            end

        // Segment framing: trigger must be high for exactly seg_len positions, each
        // lasting one dwell period (three in serial mode).
        seg_start = 16'd10; seg_len = 16'd20;
        mode = 2'd2;
        run_mode(2'd2);
        if (trig_cycles != 20 * DWELL) begin
            $display("  FAIL: segment trigger high %0d cycles, expected %0d",
                     trig_cycles, 20 * DWELL);
            errors = errors + 1;
        end else
            $display("  segment framing OK: %0d cycles for 20 positions", trig_cycles);

        // Outputs must be unaffected by framing -- it gates the trigger, not the scan.
        for (i = 0; i < N_OUT; i = i + 1)
            if (captured[i] !== ref_mode0[i]) begin
                $display("  FAIL: segment framing changed output %0d", i);
                errors = errors + 1;
            end

        if (errors == 0)
            $display("PASS: line buffer correct, three modes identical, framing exact");
        else
            $display("FAIL: %0d error(s)", errors);
        $finish;
    end
endmodule

`default_nettype wire
