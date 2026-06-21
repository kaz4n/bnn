//============================================================================
// P1 -- CW305 register module for the layer-1 BNN conv (modeled on cw305_reg_aes).
// STATUS: hardware-integration template, UNTESTED on hardware.
//
// Register map (byte-addressed via reg_address / reg_bytecnt):
//   0x0A REG_IMAGE   [784 B]  input image, row-major, pixels 0..255   (USB write)
//   0x0B REG_KERNEL  [BPK B]  one packed KxK kernel (P2 convention)    (USB write)
//   0x05 REG_GO      [1 B]    write 1 -> run conv once                 (USB write)
//   0x0E REG_STATUS  [1 B]    bit0 = done                              (USB read)
//   0x0D REG_OUTPUT  [1568 B] output feature map, int16 LE per pixel   (USB read)
//
// The conv runs on crypt_clk (the synchronously-sampled clock); trigger is asserted
// for exactly the conv (ap_start..ap_done) so CW-Lite captures only layer-1 conv power.
//============================================================================
`default_nettype none
`timescale 1ns/1ps

module cw305_reg_bnn #(
    parameter pBYTECNT_SIZE = 7,
    parameter KSIZE = 3,
    parameter BPK   = 2
)(
    input  wire        usb_clk,
    input  wire        crypt_clk,
    input  wire [7:0]  reg_address,
    input  wire [pBYTECNT_SIZE-1:0] reg_bytecnt,
    output reg  [7:0]  reg_datao,
    input  wire [7:0]  reg_datai,
    input  wire        reg_read,
    input  wire        reg_write,
    input  wire        reg_addrvalid,
    output wire        trigger,
    output wire [7:0]  leds
);
    localparam IMG = 28*28;     // 784
    localparam REG_IMAGE=8'h0A, REG_KERNEL=8'h0B, REG_GO=8'h05,
               REG_STATUS=8'h0E, REG_OUTPUT=8'h0D;

    // ---- storage (true dual-port BRAM inferred) ----
    (* ram_style="block" *) reg [7:0]  img_mem [0:IMG-1];      // usb write, crypt read
    (* ram_style="block" *) reg [7:0]  krn_mem [0:BPK-1];
    (* ram_style="block" *) reg [15:0] out_mem [0:IMG-1];      // crypt write, usb read
    reg go_usb;

    // ---- USB-domain writes ----
    always @(posedge usb_clk) begin
        if (reg_addrvalid && reg_write) begin
            case (reg_address)
                REG_IMAGE : img_mem[reg_bytecnt]  <= reg_datai;   // (extend addr for >128B)
                REG_KERNEL: krn_mem[reg_bytecnt]  <= reg_datai;
                REG_GO    : go_usb <= reg_datai[0];
                default   : ;
            endcase
        end else begin
            go_usb <= 1'b0;                       // GO is a one-shot pulse
        end
    end

    // ---- USB-domain reads ----
    wire done_usb;
    always @(*) begin
        reg_datao = 8'h00;
        if (reg_addrvalid && reg_read) begin
            case (reg_address)
                REG_STATUS: reg_datao = {7'b0, done_usb};
                REG_OUTPUT: reg_datao = reg_bytecnt[0] ? out_mem[reg_bytecnt>>1][15:8]
                                                       : out_mem[reg_bytecnt>>1][7:0];
                default   : reg_datao = 8'h00;
            endcase
        end
    end

    // ---- CDC: go pulse usb_clk -> crypt_clk ----
    (* ASYNC_REG="TRUE" *) reg go_meta, go_sync, go_prev;
    always @(posedge crypt_clk) begin
        go_meta <= go_usb; go_sync <= go_meta; go_prev <= go_sync;
    end
    wire go_crypt = go_sync & ~go_prev;          // rising edge

    // ---- bnn_conv1 HLS IP control (ap_ctrl_hs) ----
    wire ap_start = go_crypt;
    wire ap_done, ap_idle, ap_ready;
    reg  busy;
    always @(posedge crypt_clk) begin
        if (ap_start) busy <= 1'b1;
        else if (ap_done) busy <= 1'b0;
    end
    assign trigger = busy;                        // CW-Lite samples while busy

    // done flag back to usb domain
    reg done_crypt;
    always @(posedge crypt_clk) begin
        if (ap_start)      done_crypt <= 1'b0;
        else if (ap_done)  done_crypt <= 1'b1;
    end
    (* ASYNC_REG="TRUE" *) reg d0, d1;
    always @(posedge usb_clk) begin d0 <= done_crypt; d1 <= d0; end
    assign done_usb = d1;

    // HLS IP instance (ports follow Vivado HLS BRAM + ap_ctrl_hs naming; adjust to the
    // exact names in the exported IP's *_hw.h / RTL after csynth).
    wire [10:0] img_addr;  wire        img_ce;   wire [7:0]  img_q;
    wire [3:0]  krn_addr;  wire        krn_ce;   wire [7:0]  krn_q;
    wire [10:0] out_addr;  wire        out_ce, out_we; wire [15:0] out_d;

    assign img_q = img_mem[img_addr];
    assign krn_q = krn_mem[krn_addr];
    always @(posedge crypt_clk) if (out_ce & out_we) out_mem[out_addr] <= out_d;

    bnn_conv1 U_conv (
        .ap_clk(crypt_clk), .ap_rst(1'b0),
        .ap_start(ap_start), .ap_done(ap_done), .ap_idle(ap_idle), .ap_ready(ap_ready),
        .img_Addr_A(img_addr), .img_EN_A(img_ce), .img_Din_A(), .img_Dout_A(img_q),
        .img_WEN_A(), .img_Clk_A(), .img_Rst_A(),
        .packed_kern_Addr_A(krn_addr), .packed_kern_EN_A(krn_ce),
        .packed_kern_Dout_A(krn_q),
        .out_r_Addr_A(out_addr), .out_r_EN_A(out_ce), .out_r_WEN_A(out_we),
        .out_r_Din_A(out_d)
    );

    assign leds = {7'b0, busy};
endmodule
`default_nettype wire
