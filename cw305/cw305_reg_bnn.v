//============================================================================
// CW305 register module for the layer-1 BNN conv (modeled on cw305_reg_aes.v).
//
// Byte-address space = {reg_address, reg_bytecnt} (fpga_write(addr,..) -> byte addr
// addr*128+i). Regions (host writes with these python addresses to fpga_write/read):
//   IMAGE   bytes 0      .. 783     (fpga_write addr 0,  784 bytes, pixels 0..255)
//   OUTPUT  bytes 2048   .. 3615    (fpga_read  addr 16, 1568 bytes = 784 int16 LE)
//   KERNEL  bytes 4096   .. 4096+BPK-1 (fpga_write addr 32, packed KxK kernel)
//   GO      reg_address 33          (fpga_write addr 33,[1] -> run conv once)
//   STATUS  reg_address 34          (fpga_read  addr 34 -> bit0 = busy)
//   SENSOR  bytes 6144   .. 7711    (fpga_read  addr 48, 1568 bytes = 784 uint16 LE)
//   SENSOR_CTRL reg_address 35       (bit0 = RO sensor enable, default 1)
//
// Conv runs on crypto_clk (synchronously sampled). Trigger asserted only while the conv
// writes output (II=1 -> every cycle) => captured window = exactly the 784 conv cycles.
//============================================================================
`default_nettype none
`timescale 1ns/1ps
// CW305 reg interface on this board effectively caps reg_address ~34, so the high SENSOR
// region (reg 48) is unreadable. Store the RO-sensor trace in out_mem instead and read it
// at the proven-working OUTPUT region (reg 16). Conv still runs (sensor measures it); we
// just store sensor values rather than the conv output during capture.
`define SENSOR_TO_OUT
// Paper-faithful path = EXTERNAL ANALOG capture (CW-Lite scope), not the RO sensor (which
// won't load: its combinational loops break CW305 USB config). NO_RO_SENSOR removes the ROs
// -> bitstream loads; conv output stays in out_mem (func verify) and tio_trigger = busy_c
// spans the conv for the scope.
// `define NO_RO_SENSOR   // RE-ENABLED RO sensor: loads fine with BITSTREAM.GENERAL.CRC
                          // DISABLE (allow_ro_loops.tcl). Was a leftover from the external
                          // analog-capture path; kept the sensor out of every rebuild.

module cw305_reg_bnn #(
    parameter pADDR_WIDTH   = 21,
    parameter pBYTECNT_SIZE = 7,
    parameter KSIZE = 3,
    parameter BPK   = 2
)(
    input  wire                                 usb_clk,
    input  wire                                 crypto_clk,
    input  wire                                 reset_i,
    input  wire [pADDR_WIDTH-pBYTECNT_SIZE-1:0] reg_address,
    input  wire [pBYTECNT_SIZE-1:0]             reg_bytecnt,
    output reg  [7:0]                           read_data,
    input  wire [7:0]                           write_data,
    input  wire                                 reg_read,
    input  wire                                 reg_write,
    input  wire                                 reg_addrvalid,
    input  wire                                 exttrigger_in,
    output reg                                  O_user_led,
    output wire                                 leak_led,
    output wire                                 tio_trigger
);
    localparam IMG = 28*28;                 // 784
    localparam [pADDR_WIDTH-1:0] OUT_BASE = 21'd2048;
    localparam [pADDR_WIDTH-1:0] KRN_BASE = 21'd4096;
    localparam [pADDR_WIDTH-1:0] SNS_BASE = 21'd6144;
    localparam GO_ADDR = 33, STAT_ADDR = 34, SENSOR_CTRL_ADDR = 35;

    wire [pADDR_WIDTH-1:0] byteaddr = {reg_address, reg_bytecnt};

    // Plain reg arrays -> inferred true-dual-port BLOCK RAM with SYNCHRONOUS reads
    // (1-cycle latency = both the usb_reg_fe read protocol AND the HLS BRAM protocol).
    reg [7:0]  img_mem [0:IMG-1];   // port A: usb (write+host-read) | port B: crypto (conv read)
    reg [7:0]  krn_mem [0:127];     // port A: usb write           | port B: crypto (conv read)
    reg [15:0] out_mem [0:IMG-1];   // port A: crypto write        | port B: usb (host read)
    reg [15:0] sensor_mem [0:IMG-1]; // port A: crypto write       | port B: usb (host read)

    wire busy_usb;
    wire [pADDR_WIDTH-1:0] outoff = byteaddr - OUT_BASE;
    wire [pADDR_WIDTH-1:0] snsoff = byteaddr - SNS_BASE;
    reg sensor_enable_usb = 1'b1;

    //------------- USB domain: writes + registered (1-cycle) reads -------------
    reg go_pulse_usb;
    always @(posedge usb_clk) begin
        go_pulse_usb <= 1'b0;
        if (reset_i) begin
            O_user_led <= 1'b0;
            sensor_enable_usb <= 1'b1;
        end
        // writes
        if (reg_addrvalid && reg_write) begin
            if (byteaddr < IMG)
                img_mem[byteaddr[9:0]] <= write_data;
            else if (byteaddr >= KRN_BASE && byteaddr < KRN_BASE + 128)
                krn_mem[byteaddr[6:0]] <= write_data;
            else if (reg_address == GO_ADDR)
                go_pulse_usb <= 1'b1;
            else if (reg_address == SENSOR_CTRL_ADDR)
                sensor_enable_usb <= write_data[0];
            else if (reg_address == 14'h01)
                O_user_led <= write_data[0];
        end
        // registered read (data valid 1 clk after byteaddr presented, as protocol requires)
        if (byteaddr < IMG)
            read_data <= img_mem[byteaddr[9:0]];                 // image readback (debug)
        else if (byteaddr >= OUT_BASE && byteaddr < OUT_BASE + 2*IMG)
            read_data <= outoff[0] ? out_mem[outoff[10:1]][15:8]
                                   : out_mem[outoff[10:1]][7:0];
        else if (byteaddr >= SNS_BASE && byteaddr < SNS_BASE + 2*IMG)
            read_data <= 8'hC3;   // DIAGNOSTIC constant: SNS branch reachability
        else if (reg_address == STAT_ADDR)
            read_data <= busy_usb ? 8'h01 : 8'h00;        // bit0 is the documented busy flag
        else if (reg_address == SENSOR_CTRL_ADDR)
            read_data <= 8'h5A;   // DIAGNOSTIC constant: CTRL branch reachability
        else
            read_data <= 8'h00;
    end

    //------------------------------------------------------------------ go -> crypto
    wire go_crypt;
    cdc_pulse U_go (.reset_i(reset_i), .src_clk(usb_clk), .src_pulse(go_pulse_usb),
                    .dst_clk(crypto_clk), .dst_pulse(go_crypt));
    // hardware trigger (TIO) can also launch, like the AES example
    (* ASYNC_REG="TRUE" *) reg [2:0] extt;
    always @(posedge crypto_clk)
        if (reset_i) extt <= 3'b000;
        else extt <= {extt[1:0], exttrigger_in};

    //------------------------------------------------------------------ HLS conv IP
    wire ap_done, ap_idle, ap_ready;
    reg  busy_c;
    wire start_req = go_crypt | (extt[1] & ~extt[2]);
    wire ap_start = start_req & ~busy_c;          // reject overlapping launches
    always @(posedge crypto_clk)
        if (reset_i) busy_c <= 1'b0;
        else if (ap_start) busy_c <= 1'b1;
        else if (ap_done) busy_c <= 1'b0;

    // HLS emits 32-bit BYTE addresses: img/kern 8-bit (elem = addr), out 16-bit (elem = addr>>1).
    wire [31:0] img_addr; wire img_ce; wire [7:0] img_q;
    wire [31:0] krn_addr; wire krn_ce; wire [7:0] krn_q;
    wire [31:0] out_addr; wire out_ce; wire [1:0] out_we; wire [15:0] out_d;
    wire [9:0] out_index = out_addr[10:1];
    // conv-side reads registered on crypto_clk -> latency 1, matches HLS BRAM protocol
    reg [7:0] img_q_r, krn_q_r;
    always @(posedge crypto_clk) begin
        img_q_r <= img_mem[img_addr[9:0]];
        krn_q_r <= krn_mem[krn_addr[6:0]];
    end
    assign img_q = img_q_r;
    assign krn_q = krn_q_r;
`ifdef NO_RO_SENSOR
    always @(posedge crypto_clk) if (out_ce && (|out_we))
        out_mem[out_addr[10:1]] <= out_d;        // RO removed -> store conv output
`else
    always @(posedge crypto_clk) if (out_ce && (|out_we))
        out_mem[out_addr[10:1]] <= sensor_sample;
`endif

    // Keep v1 deterministic: the RO sensor is enabled by default for every run.
    // Use a preserved flop instead of a constant so Vivado keeps the RO gate sane.
    (* DONT_TOUCH = "true" *) reg sensor_enable_c = 1'b1;
    always @(posedge crypto_clk) begin
        if (reset_i)
            sensor_enable_c <= 1'b1;
        else
            sensor_enable_c <= 1'b1;
    end
    wire sensor_sample_en = sensor_enable_c && out_ce && (|out_we) && (out_index < IMG);
    wire [15:0] sensor_sample;
`ifdef NO_RO_SENSOR
    assign sensor_sample = 16'd0;            // RO sensor removed for the load test
`else
    // Gate RO oscillation to the conv run only (busy_c). Static at config -> bitstream
    // loads (active comb-loop ROs otherwise break the CW305 USB configuration/readback).
    ro_counter_sensor #(.N_RO(16), .COUNTER_BITS(16), .SAMPLE_BITS(16)) U_ro_sensor (
        .clk(crypto_clk),
        .reset(reset_i),
        .enable(busy_c),
        .sample_en(sensor_sample_en),
        .sample_value(sensor_sample)
    );
    always @(posedge crypto_clk) if (sensor_sample_en) sensor_mem[out_index] <= sensor_sample;
`endif

    wire [7:0] leak_byte;
    bnn_conv1 U_conv (
        .ap_clk(crypto_clk), .ap_rst(reset_i),
        .ap_start(ap_start), .ap_done(ap_done), .ap_idle(ap_idle), .ap_ready(ap_ready),
        .img_Addr_A(img_addr), .img_EN_A(img_ce), .img_WEN_A(), .img_Din_A(),
        .img_Dout_A(img_q), .img_Clk_A(), .img_Rst_A(),
        .packed_kern_Addr_A(krn_addr), .packed_kern_EN_A(krn_ce), .packed_kern_WEN_A(),
        .packed_kern_Din_A(), .packed_kern_Dout_A(krn_q), .packed_kern_Clk_A(), .packed_kern_Rst_A(),
        .out_r_Addr_A(out_addr), .out_r_EN_A(out_ce), .out_r_WEN_A(out_we),
        .out_r_Din_A(out_d), .out_r_Dout_A(16'b0), .out_r_Clk_A(), .out_r_Rst_A(),
        .leak(leak_byte)
    );
    // route the leak sink to a top-level pin so Vivado keeps the FANOUT copies AND the conv
    // MAC output (out_d) -- otherwise out_d is unused (sensor goes to out_mem) and Vivado
    // would prune the data-dependent MAC the sensor needs to observe.
    reg leak_r;
    always @(posedge crypto_clk)
        if (reset_i) leak_r <= 1'b0;
        else leak_r <= ^leak_byte ^ (^out_d);
    assign leak_led = leak_r;

    // HLS BRAM addresses are byte addresses (data width 8/16) -> word index = addr>>(log2 width bytes)
    // img/kern are 8-bit (Addr in bytes = element index). out is 16-bit (Addr in bytes -> /2).
    // Vivado HLS 2016.4 emits element-indexed addresses here, so use directly.

    // Trigger continuously from the first output write through the last output write.
    // Output-valid itself has K-1 low clocks between rows, so directly routing it would
    // produce a pulse train and a 784-clock capture would be truncated (K=3 needs an
    // inclusive 838-clock first-to-last span). The host still stores cycle0_offset so
    // extraction maps physical clocks to the output grid.
    reg trig_r;
    always @(posedge crypto_clk) begin
        if (reset_i || ap_start)
            trig_r <= 1'b0;
        else if (out_ce && (|out_we) && (out_index == 0))
            trig_r <= 1'b1;
        else if (out_ce && (|out_we) && (out_index == IMG-1))
            trig_r <= 1'b0;
    end
    assign tio_trigger = trig_r;

    // busy back to usb domain
    (* ASYNC_REG="TRUE" *) reg [1:0] busy_sync;
    always @(posedge usb_clk)
        if (reset_i) busy_sync <= 2'b00;
        else busy_sync <= {busy_sync[0], busy_c};
    assign busy_usb = busy_sync[1];
endmodule
`default_nettype wire
