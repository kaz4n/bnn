`timescale 1ns/1ps
`default_nettype none

// CW305 top level for the RGB leakage study.
//
// Identical in structure to cw305_leakage_grey_top.sv; it instantiates
// cw305_reg_linebuf_rgb instead of the grey register file and carries IMG_SIDE instead
// of LEAK_LANES, because this design has no leakage amplifier to size.
//
// The register file streams pixels from block RAM to a line-buffer core. An earlier
// revision fed 27 random-access taps from distributed RAM and did not fit (115% LUTs);
// see cw305_reg_linebuf_rgb.sv.
module cw305_leakage_rgb_top #(
    parameter integer pBYTECNT_SIZE = 7,
    parameter integer pADDR_WIDTH   = 21,
    parameter integer DWELL         = 8,
    parameter integer IMG_SIDE      = 32,
    // Matches the grey builds: crypto clock from the CW-Lite via tio_clkin.
    parameter [4:0] CLOCK_SETTINGS  = 5'b01101
)(
    input  wire                   usb_clk,
    inout  wire [7:0]             usb_data,
    input  wire [pADDR_WIDTH-1:0] usb_addr,
    input  wire                   usb_rdn,
    input  wire                   usb_wrn,
    input  wire                   usb_cen,
    input  wire                   usb_trigger,
    input  wire                   j16_sel,
    input  wire                   k16_sel,
    input  wire                   k15_sel,
    input  wire                   l14_sel,
    input  wire                   pushbutton,
    output wire                   led1,
    output wire                   led2,
    output wire                   led3,
    input  wire                   pll_clk1,
    output wire                   tio_trigger,
    output wire                   tio_clkout,
    input  wire                   tio_clkin
);
    wire usb_clk_buf;
    wire [7:0] usb_dout;
    wire isout;
    assign usb_data = isout ? usb_dout : 8'bz;

    wire [pADDR_WIDTH-pBYTECNT_SIZE-1:0] reg_address;
    wire [pBYTECNT_SIZE-1:0] reg_bytecnt;
    wire [7:0] write_data;
    wire [7:0] read_data;
    wire reg_read, reg_write, reg_addrvalid;
    wire [4:0] clk_settings = CLOCK_SETTINGS;
    wire crypto_clk;
    wire reset = !pushbutton;
    wire leak_led;

    assign led1 = leak_led;
    assign led2 = 1'b0;

    cw305_usb_reg_fe #(.pBYTECNT_SIZE(pBYTECNT_SIZE), .pADDR_WIDTH(pADDR_WIDTH))
    U_usb_reg_fe (
        .rst(reset), .usb_clk(usb_clk_buf), .usb_din(usb_data), .usb_dout(usb_dout),
        .usb_rdn(usb_rdn), .usb_wrn(usb_wrn), .usb_cen(usb_cen), .usb_alen(1'b0),
        .usb_addr(usb_addr), .usb_isout(isout), .reg_address(reg_address),
        .reg_bytecnt(reg_bytecnt), .reg_datao(write_data), .reg_datai(read_data),
        .reg_read(reg_read), .reg_write(reg_write), .reg_addrvalid(reg_addrvalid)
    );

    cw305_reg_linebuf_rgb #(
        .pADDR_WIDTH(pADDR_WIDTH), .pBYTECNT_SIZE(pBYTECNT_SIZE),
        .DWELL(DWELL), .IMG_SIDE(IMG_SIDE)
    ) U_reg (
        .usb_clk(usb_clk_buf), .crypto_clk(crypto_clk), .reset_i(reset),
        .reg_address(reg_address), .reg_bytecnt(reg_bytecnt), .read_data(read_data),
        .write_data(write_data), .reg_read(reg_read), .reg_write(reg_write),
        .reg_addrvalid(reg_addrvalid), .exttrigger_in(usb_trigger),
        .O_user_led(led3), .leak_led(leak_led), .tio_trigger(tio_trigger)
    );

    clocks U_clocks (
        .usb_clk(usb_clk), .usb_clk_buf(usb_clk_buf),
        .I_j16_sel(j16_sel), .I_k16_sel(k16_sel), .I_clock_reg(clk_settings),
        .I_cw_clkin(tio_clkin), .I_pll_clk1(pll_clk1),
        .O_cw_clkout(tio_clkout), .O_cryptoclk(crypto_clk)
    );
endmodule

`default_nettype wire
