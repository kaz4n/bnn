//============================================================================
// CW305 top for the layer-1 BNN conv. Minimal edit of NewAE's stock cw305_top.v:
// same ports / usb_reg_fe / clocks (so stock cw305.xdc applies unchanged) -- only the
// AES core (cw305_reg_aes + aes_core) is replaced by cw305_reg_bnn + the HLS bnn_conv1.
//============================================================================
`timescale 1ns / 1ps
`default_nettype none

module cw305_bnn_top #(
    parameter pBYTECNT_SIZE = 7,
    parameter pADDR_WIDTH   = 21,
    parameter KSIZE = 3,
    parameter BPK   = (KSIZE*KSIZE+7)/8,
    // Current bench default:
    //   bits [2:0] = 3'b101 -> crypto clock = 20-pin CW-Lite clock input
    //   bits [4:3] = 2'b01  -> export crypto clock on TIO clockout
    parameter [4:0] CLOCK_SETTINGS = 5'b01101
)(
    // USB Interface
    input  wire                   usb_clk,
    inout  wire [7:0]             usb_data,
    input  wire [pADDR_WIDTH-1:0] usb_addr,
    input  wire                   usb_rdn,
    input  wire                   usb_wrn,
    input  wire                   usb_cen,
    input  wire                   usb_trigger,
    // DIP switches / button / LEDs
    input  wire                   j16_sel,
    input  wire                   k16_sel,
    input  wire                   k15_sel,
    input  wire                   l14_sel,
    input  wire                   pushbutton,
    output wire                   led1,
    output wire                   led2,
    output wire                   led3,
    // PLL
    input  wire                   pll_clk1,
    // 20-pin connector
    output wire                   tio_trigger,
    output wire                   tio_clkout,
    input  wire                   tio_clkin
);
    wire usb_clk_buf;
    wire [7:0] usb_dout;
    wire isout;
    assign usb_data = isout ? usb_dout : 8'bZ;

    wire [pADDR_WIDTH-pBYTECNT_SIZE-1:0] reg_address;
    wire [pBYTECNT_SIZE-1:0]             reg_bytecnt;
    wire [7:0]  write_data, read_data;
    wire        reg_read, reg_write, reg_addrvalid;
    wire [4:0]  clk_settings = CLOCK_SETTINGS;
    wire        crypt_clk;

    wire resetn = pushbutton;
    wire reset  = !resetn;

    // heartbeats removed (data-independent switching hurts SNR). led1 carries the conv
    // leak-sink (keeps the FANOUT leakage copies from being pruned); led2 off.
    wire leak_led;
    assign led1 = leak_led;
    assign led2 = 1'b0;

    cw305_usb_reg_fe #(.pBYTECNT_SIZE(pBYTECNT_SIZE), .pADDR_WIDTH(pADDR_WIDTH))
    U_usb_reg_fe (
        .rst(reset), .usb_clk(usb_clk_buf), .usb_din(usb_data), .usb_dout(usb_dout),
        .usb_rdn(usb_rdn), .usb_wrn(usb_wrn), .usb_cen(usb_cen), .usb_alen(1'b0),
        .usb_addr(usb_addr), .usb_isout(isout),
        .reg_address(reg_address), .reg_bytecnt(reg_bytecnt),
        .reg_datao(write_data), .reg_datai(read_data),
        .reg_read(reg_read), .reg_write(reg_write), .reg_addrvalid(reg_addrvalid)
    );

    cw305_reg_bnn #(.pADDR_WIDTH(pADDR_WIDTH), .pBYTECNT_SIZE(pBYTECNT_SIZE),
                    .KSIZE(KSIZE), .BPK(BPK))
    U_reg_bnn (
        .usb_clk(usb_clk_buf), .crypto_clk(crypt_clk), .reset_i(reset),
        .reg_address(reg_address), .reg_bytecnt(reg_bytecnt),
        .read_data(read_data), .write_data(write_data),
        .reg_read(reg_read), .reg_write(reg_write), .reg_addrvalid(reg_addrvalid),
        .exttrigger_in(usb_trigger), .O_user_led(led3),
        .leak_led(leak_led), .tio_trigger(tio_trigger)
    );

    clocks U_clocks (
        .usb_clk(usb_clk), .usb_clk_buf(usb_clk_buf),
        .I_j16_sel(j16_sel), .I_k16_sel(k16_sel), .I_clock_reg(clk_settings),
        .I_cw_clkin(tio_clkin), .I_pll_clk1(pll_clk1),
        .O_cw_clkout(tio_clkout), .O_cryptoclk(crypt_clk)
    );
endmodule
`default_nettype wire
