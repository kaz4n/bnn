//============================================================================
// P1 -- CW305 top-level wrapping the layer-1 BNN conv IP (the attack target).
//
// STATUS: hardware-integration template. Compiles conceptually against the stock
// ChipWhisperer CW305 reference, but is UNTESTED on hardware (you have no bench yet).
// You MUST add these stock files from the ChipWhisperer repo to the Vivado project
// (hardware/victims/cw305_artixtarget/fpga/common/):
//     cw305_usb_reg_fe.v      USB register frontend (SAM3U <-> fabric)
//     cw305_reg_aes.v         (reference only -- our cw305_reg_bnn below replaces it)
//     cw305_pll.v / clocks    target clock generation
//     cw305_main.xdc          pin constraints (reuse as-is; ports below match it)
// and the HLS-exported IP `bnn_conv1` (see ../build/run_hls.tcl, run_vivado.tcl).
//
// Data flow: host writes 28x28 image + one packed KxK kernel over USB into BRAM ->
// host writes GO -> FSM runs bnn_conv1 on the target clock (the clock CW-Lite samples
// synchronously) -> trigger asserted for exactly the conv -> host reads back output.
//============================================================================
`default_nettype none
`timescale 1ns/1ps

module cw305_bnn_top #(
    parameter pADDR_WIDTH = 21,
    parameter pBYTECNT_SIZE = 7,
    parameter KSIZE = 3,                 // 3 = Model 1, 5 = Model 2
    parameter BPK   = (KSIZE*KSIZE+7)/8  // bytes per packed kernel (2 or 4)
)(
    // ---- USB / SAM3U interface (ports match stock cw305_main.xdc) ----
    input  wire        usb_clk,
    inout  wire [7:0]  usb_data,
    input  wire [pADDR_WIDTH-1:0] usb_addr,
    input  wire        usb_rdn,
    input  wire        usb_wrn,
    input  wire        usb_cen,
    input  wire        usb_trigger,
    // ---- clock source pins ----
    input  wire        pll_clk1,         // on-board PLL (target clock source)
    // ---- side-channel / IO ----
    output wire        tio_trigger,      // -> CW-Lite GPIO4 (capture trigger)
    output wire        tio_clkout,       // -> CW-Lite HS-In (synchronous sample clock)
    output wire [7:0]  leds
);
    //--------------------------------------------------------------------
    // Clocks. crypt_clk is the BNN working clock AND the clock CW-Lite samples
    // against (synchronous capture). Keep it low (~10-25 MHz) -- see SETUP_PLAN.md C3.
    //--------------------------------------------------------------------
    wire crypt_clk = pll_clk1;           // route a low PLL freq here in cw305_pll
    assign tio_clkout = crypt_clk;       // feed CW-Lite HS-In for synchronous capture

    //--------------------------------------------------------------------
    // USB register frontend (STOCK module -- add cw305_usb_reg_fe.v to project)
    //--------------------------------------------------------------------
    wire [7:0]  reg_address;
    wire [pBYTECNT_SIZE-1:0] reg_bytecnt;
    wire [7:0]  reg_datao;        // BNN -> USB
    wire [7:0]  reg_datai;        // USB -> BNN
    wire        reg_read, reg_write, reg_addrvalid;

    cw305_usb_reg_fe #(
        .pADDR_WIDTH(pADDR_WIDTH), .pBYTECNT_SIZE(pBYTECNT_SIZE)
    ) U_usb_reg_fe (
        .usb_clk(usb_clk), .usb_din(usb_data), .usb_dout(),  // wire per stock template
        .usb_addr(usb_addr), .usb_rdn(usb_rdn), .usb_wrn(usb_wrn), .usb_cen(usb_cen),
        .reg_address(reg_address), .reg_bytecnt(reg_bytecnt),
        .reg_datao(reg_datao), .reg_datai(reg_datai),
        .reg_read(reg_read), .reg_write(reg_write), .reg_addrvalid(reg_addrvalid)
    );

    //--------------------------------------------------------------------
    // BNN register module: image/kernel/output BRAM, GO/STATUS, conv IP, trigger.
    //--------------------------------------------------------------------
    cw305_reg_bnn #(
        .pBYTECNT_SIZE(pBYTECNT_SIZE), .KSIZE(KSIZE), .BPK(BPK)
    ) U_reg_bnn (
        .usb_clk(usb_clk), .crypt_clk(crypt_clk),
        .reg_address(reg_address), .reg_bytecnt(reg_bytecnt),
        .reg_datao(reg_datao), .reg_datai(reg_datai),
        .reg_read(reg_read), .reg_write(reg_write), .reg_addrvalid(reg_addrvalid),
        .trigger(tio_trigger), .leds(leds)
    );
endmodule
`default_nettype wire
