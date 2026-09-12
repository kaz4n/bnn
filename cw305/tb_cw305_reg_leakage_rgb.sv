`timescale 1ns/1ps
`default_nettype none

// Integration testbench for cw305_reg_leakage_rgb + rgb_leakage_core.
//
// Exercises the real USB register path: every image byte, the 27-bit kernel and the mode
// are written through reg_address/reg_bytecnt exactly as the host driver will write them,
// then GO is pulsed and the outputs are read back from the output page and compared
// against an independent golden model.
//
// The 27-tap window wiring in the register file is the highest-risk part of this design:
// a transposed channel or a wrong row stride produces a plausible-looking but wrong
// result that no amount of host-side analysis would catch. This testbench is what
// catches it, by checking every output of every mode against arithmetic computed here.
//
// IMG_SIDE is 8 rather than 32 so the run is fast; the plane decode is derived from the
// parameter, so the logic under test is identical to the synthesised build.
module tb_cw305_reg_leakage_rgb;
    localparam integer pADDR_WIDTH   = 21;
    localparam integer pBYTECNT_SIZE = 7;
    localparam integer DWELL         = 4;
    localparam integer IMG_SIDE      = 8;

    localparam integer PLANE    = IMG_SIDE * IMG_SIDE;      // 64
    localparam integer N_IMAGE  = PLANE * 3;                // 192
    localparam integer OUT_SIDE = IMG_SIDE - 2;             // 6
    localparam integer N_OUTPUT = OUT_SIDE * OUT_SIDE;      // 36

    localparam [pADDR_WIDTH-1:0] OUT_BASE = 21'd8192;
    localparam [pADDR_WIDTH-1:0] KRN_BASE = 21'd16384;
    localparam integer GO_ADDR   = 33;
    localparam integer STAT_ADDR = 34;
    localparam integer SIG_ADDR  = 35;

    reg usb_clk = 1'b0, crypto_clk = 1'b0, reset_i = 1'b1;
    reg [pADDR_WIDTH-pBYTECNT_SIZE-1:0] reg_address = 0;
    reg [pBYTECNT_SIZE-1:0] reg_bytecnt = 0;
    reg [7:0] write_data = 8'h00;
    reg reg_read = 1'b0, reg_write = 1'b0, reg_addrvalid = 1'b0;
    reg exttrigger_in = 1'b0;
    wire [7:0] read_data;
    wire O_user_led, leak_led, tio_trigger;

    // Deliberately unrelated clocks: the design crosses usb_clk to crypto_clk and the
    // testbench should not accidentally make that crossing synchronous.
    always #5   usb_clk    = ~usb_clk;      // 100 MHz
    always #16  crypto_clk = ~crypto_clk;   // ~31 MHz

    cw305_reg_leakage_rgb #(
        .pADDR_WIDTH(pADDR_WIDTH), .pBYTECNT_SIZE(pBYTECNT_SIZE),
        .DWELL(DWELL), .IMG_SIDE(IMG_SIDE)
    ) dut (
        .usb_clk(usb_clk), .crypto_clk(crypto_clk), .reset_i(reset_i),
        .reg_address(reg_address), .reg_bytecnt(reg_bytecnt),
        .read_data(read_data), .write_data(write_data),
        .reg_read(reg_read), .reg_write(reg_write), .reg_addrvalid(reg_addrvalid),
        .exttrigger_in(exttrigger_in), .O_user_led(O_user_led),
        .leak_led(leak_led), .tio_trigger(tio_trigger)
    );

    reg [7:0] img [0:N_IMAGE-1];
    reg [26:0] kern;
    integer errors;
    integer i, m;

    task usb_write(input [pADDR_WIDTH-1:0] addr, input [7:0] d);
        begin
            @(negedge usb_clk);
            reg_address   = addr[pADDR_WIDTH-1:pBYTECNT_SIZE];
            reg_bytecnt   = addr[pBYTECNT_SIZE-1:0];
            write_data    = d;
            reg_addrvalid = 1'b1;
            reg_write     = 1'b1;
            @(negedge usb_clk);
            reg_write     = 1'b0;
            reg_addrvalid = 1'b0;
        end
    endtask

    task usb_read(input [pADDR_WIDTH-1:0] addr, output [7:0] d);
        begin
            @(negedge usb_clk);
            reg_address   = addr[pADDR_WIDTH-1:pBYTECNT_SIZE];
            reg_bytecnt   = addr[pBYTECNT_SIZE-1:0];
            reg_addrvalid = 1'b1;
            reg_read      = 1'b1;
            @(posedge usb_clk);          // read_data is registered on usb_clk
            @(negedge usb_clk);
            d             = read_data;
            reg_read      = 1'b0;
            reg_addrvalid = 1'b0;
        end
    endtask

    // Golden convolution, computed from the testbench's own copy of the image.
    function automatic signed [15:0] golden(input integer win);
        integer gr, gc, c, t, rr, cc, idx;
        reg signed [15:0] s;
        begin
            gr = win / OUT_SIDE;
            gc = win % OUT_SIDE;
            s = 16'sd0;
            for (c = 0; c < 3; c = c + 1)
                for (t = 0; t < 9; t = t + 1) begin
                    rr  = gr + (t / 3);
                    cc  = gc + (t % 3);
                    idx = c*PLANE + rr*IMG_SIDE + cc;
                    s = kern[c*9+t] ? (s + $signed({8'd0, img[idx]}))
                                    : (s - $signed({8'd0, img[idx]}));
                end
            golden = s;
        end
    endfunction

    reg [7:0] rb_lo, rb_hi, st;
    reg signed [15:0] got;
    integer w, guard;

    task run_mode(input [1:0] mode_sel);
        begin
            usb_write(KRN_BASE + 4, {6'b0, mode_sel});

            // GO: any write inside the GO page pulses start.
            usb_write({GO_ADDR[13:0], 7'd0}, 8'h01);

            // Wait for busy to assert then clear. Status byte 0 is busy, as in the grey
            // design, so existing host probes keep working.
            guard = 0;
            st = 8'h00;
            while (st[0] !== 1'b1 && guard < 2000) begin
                usb_read({STAT_ADDR[13:0], 7'd0}, st);
                guard = guard + 1;
            end
            if (guard >= 2000) begin
                $display("  FAIL mode %0d: busy never asserted", mode_sel);
                errors = errors + 1;
                disable run_mode;
            end
            guard = 0;
            while (st[0] !== 1'b0 && guard < 200000) begin
                usb_read({STAT_ADDR[13:0], 7'd0}, st);
                guard = guard + 1;
            end
            if (guard >= 200000) begin
                $display("  FAIL mode %0d: busy never cleared", mode_sel);
                errors = errors + 1;
                disable run_mode;
            end

            for (w = 0; w < N_OUTPUT; w = w + 1) begin
                usb_read(OUT_BASE + 2*w,     rb_lo);
                usb_read(OUT_BASE + 2*w + 1, rb_hi);
                got = $signed({rb_hi, rb_lo});
                if (got !== golden(w)) begin
                    if (errors < 10)
                        $display("  FAIL mode %0d win %0d: got %0d expected %0d",
                                 mode_sel, w, got, golden(w));
                    errors = errors + 1;
                end
            end
            $display("  mode %0d: %0d outputs checked", mode_sel, N_OUTPUT);
        end
    endtask

    integer seed;
    reg [7:0] sig [0:7];
    initial begin
        errors = 0;
        seed   = 32'h0BADF00D;

        for (i = 0; i < N_IMAGE; i = i + 1) img[i] = $random(seed) & 8'hFF;
        kern = {$random(seed)} & 27'h7FF_FFFF;

        repeat (6) @(negedge usb_clk);
        reset_i = 1'b0;
        repeat (4) @(negedge usb_clk);

        $display("RGB register file integration (IMG_SIDE=%0d, DWELL=%0d)", IMG_SIDE, DWELL);

        // Design signature must be readable before anything else is trusted.
        for (i = 0; i < 8; i = i + 1) usb_read({SIG_ADDR[13:0], i[6:0]}, sig[i]);
        if ({sig[0],sig[1],sig[2],sig[3],sig[4],sig[5],sig[6],sig[7]} != "RGBSCA01") begin
            $display("  FAIL: signature page reads %0s, expected RGBSCA01",
                     {sig[0],sig[1],sig[2],sig[3],sig[4],sig[5],sig[6],sig[7]});
            errors = errors + 1;
        end else
            $display("  signature OK: RGBSCA01");

        for (i = 0; i < N_IMAGE; i = i + 1) usb_write(i[pADDR_WIDTH-1:0], img[i]);
        usb_write(KRN_BASE + 0, kern[7:0]);
        usb_write(KRN_BASE + 1, kern[15:8]);
        usb_write(KRN_BASE + 2, kern[23:16]);
        usb_write(KRN_BASE + 3, {5'b0, kern[26:24]});

        // Image readback must round-trip, or the plane decode is wrong.
        for (i = 0; i < N_IMAGE; i = i + 16) begin
            usb_read(i[pADDR_WIDTH-1:0], rb_lo);
            if (rb_lo !== img[i]) begin
                $display("  FAIL: image readback at %0d got %0d expected %0d",
                         i, rb_lo, img[i]);
                errors = errors + 1;
            end
        end

        run_mode(2'd0);
        run_mode(2'd1);
        run_mode(2'd2);

        if (errors == 0)
            $display("PASS: signature, image round-trip and all three modes correct");
        else
            $display("FAIL: %0d error(s)", errors);
        $finish;
    end
endmodule

`default_nettype wire
