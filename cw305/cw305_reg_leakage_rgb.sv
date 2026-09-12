`timescale 1ns/1ps
`default_nettype none

// RGB register file for the CW305 leakage study.
//
// Three-channel counterpart of cw305_reg_leakage_grey.sv. Differences, and only these:
//   * the image store holds 32x32x3 planar RGB8 (3072 bytes) instead of 784 grey bytes;
//   * the kernel is 27 bits (one +/-1 weight per channel and tap) instead of 9;
//   * a MODE register selects the channel dataflow at runtime;
//   * a fixed design signature page lets the host prove which bitstream is live.
// The status page keeps byte 0 = busy so existing host probes still work.
//
// IMAGE STORE: THREE ARRAYS, NOT ONE
// ----------------------------------
// The grey design learned that a packed vector costs a 784-way one-hot write enable and
// a 784-to-1 readback mux -- 26k LUTs, and it broke the 100 MHz usb_clk constraint. An
// array infers distributed RAM instead: one write port, and each convolution tap becomes
// a RAM read.
//
// Scaling that to RGB needs care. Distributed RAM gives one write and one read port, so
// N read ports replicate the array N times. A single 3072-byte array with 27 taps would
// be 27 x 3072 = 82,944 bytes of LUTRAM, which does not fit.
//
// But every tap reads exactly ONE channel plane, and a plane is 32*32 = 1024 bytes -- a
// power of two. So the store is three 1 KB arrays with nine read ports each:
// 3 x 9 x 1024 = 27,648 bytes replicated, about 3.9x the grey design rather than 27x.
// Channel select is byteaddr[11:10] and the in-plane offset is byteaddr[9:0], both free.
//
// Wire format is planar CHW: the host sends channel 0's 1024 bytes, then channel 1, then
// channel 2. That matches numpy's (C,H,W) ordering, so the host does no repacking.
module cw305_reg_leakage_rgb #(
    parameter integer pADDR_WIDTH   = 21,
    parameter integer pBYTECNT_SIZE = 7,
    parameter integer DWELL         = 8,
    parameter integer IMG_SIDE      = 32
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
    localparam integer PLANE      = IMG_SIDE * IMG_SIDE;          // 1024
    localparam integer N_IMAGE    = PLANE * 3;                    // 3072
    localparam integer OUT_SIDE   = IMG_SIDE - 2;                 // 30
    localparam integer N_OUTPUT   = OUT_SIDE * OUT_SIDE;          // 900
    localparam integer N_OUT_BYTE = N_OUTPUT * 2;                 // 1800

    // Memory map. GO/STAT/SIG are decoded by reg_address, which is byteaddr >> 7, so a
    // data page must not overlap their 128-byte windows. The grey design's OUT_BASE of
    // 2048 cannot simply be scaled here: 4096 would span reg_address 32..46 and collide
    // with GO(33), STAT(34) and SIG(35). The pages are therefore separated explicitly.
    //
    //   image  0      .. 3071   reg_address  0.. 23
    //   GO            byteaddr >> 7 == 33
    //   STAT                        == 34
    //   SIG                         == 35
    //   out    8192   .. 9991   reg_address 64.. 78
    //   kernel 16384  .. 16388  reg_address 128
    localparam [pADDR_WIDTH-1:0] OUT_BASE = 21'd8192;             // .. 9991
    localparam [pADDR_WIDTH-1:0] KRN_BASE = 21'd16384;            // 4 kernel bytes + mode
    localparam integer GO_ADDR   = 33;
    localparam integer STAT_ADDR = 34;
    localparam integer SIG_ADDR  = 35;

    wire [pADDR_WIDTH-1:0] byteaddr = {reg_address, reg_bytecnt};
    wire [pADDR_WIDTH-1:0] outoff   = byteaddr - OUT_BASE;

    // Plane select. A plane is IMG_SIDE^2 bytes; when IMG_SIDE is a power of two so is
    // the plane, and the split is free wiring rather than a divider. Derived from the
    // parameter rather than hardcoded to 1024, so a small IMG_SIDE can be simulated
    // exhaustively and the synthesised 32x32 build uses the identical decode logic.
    localparam integer PLANE_W = $clog2(PLANE);

    wire in_image = (byteaddr < N_IMAGE);
    wire [1:0]         img_chan = byteaddr[PLANE_W+1 : PLANE_W];
    wire [PLANE_W-1:0] img_off  = byteaddr[PLANE_W-1 : 0];

    reg [7:0] image_r [0:PLANE-1];
    reg [7:0] image_g [0:PLANE-1];
    reg [7:0] image_b [0:PLANE-1];

    reg [26:0] kernel_bits;
    reg [1:0]  mode_reg;
    reg signed [15:0] out_mem [0:N_OUTPUT-1];
    reg go_pulse_usb;
    reg [7:0] go_count_usb;

    reg [23:0] crypto_heartbeat;
    reg [7:0] start_count_crypto;
    reg [15:0] write_count_crypto;
    reg [9:0] last_out_addr_crypto;
    (* ASYNC_REG="TRUE" *) reg [23:0] heartbeat_usb;
    (* ASYNC_REG="TRUE" *) reg [7:0] start_count_usb;
    (* ASYNC_REG="TRUE" *) reg [15:0] write_count_usb;
    (* ASYNC_REG="TRUE" *) reg [9:0] last_out_addr_usb;

    wire [9:0] out_index = outoff[10:1];
    wire signed [15:0] out_word = out_mem[out_index];

    // Readback of the image store, for the host's write-sanity check.
    wire [7:0] img_read = (img_chan == 2'd0) ? image_r[img_off] :
                          (img_chan == 2'd1) ? image_g[img_off] : image_b[img_off];

    wire busy_usb;
    always @(posedge usb_clk) begin
        go_pulse_usb <= 1'b0;
        if (reset_i) begin
            O_user_led   <= 1'b0;
            kernel_bits  <= 27'b0;
            mode_reg     <= 2'd0;
            go_count_usb <= 8'd0;
        end else if (reg_addrvalid && reg_write) begin
            if (in_image) begin
                if (img_chan == 2'd0)      image_r[img_off] <= write_data;
                else if (img_chan == 2'd1) image_g[img_off] <= write_data;
                else                       image_b[img_off] <= write_data;
            end
            else if (byteaddr == KRN_BASE + 0) kernel_bits[7:0]   <= write_data;
            else if (byteaddr == KRN_BASE + 1) kernel_bits[15:8]  <= write_data;
            else if (byteaddr == KRN_BASE + 2) kernel_bits[23:16] <= write_data;
            else if (byteaddr == KRN_BASE + 3) kernel_bits[26:24] <= write_data[2:0];
            else if (byteaddr == KRN_BASE + 4) mode_reg           <= write_data[1:0];
            else if (reg_address == GO_ADDR) begin
                go_pulse_usb <= 1'b1;
                go_count_usb <= go_count_usb + 8'd1;
            end
            else if (reg_address == 14'h01)
                O_user_led <= write_data[0];
        end

        heartbeat_usb     <= crypto_heartbeat;
        start_count_usb   <= start_count_crypto;
        write_count_usb   <= write_count_crypto;
        last_out_addr_usb <= last_out_addr_crypto;

        if (in_image)
            read_data <= img_read;
        else if (byteaddr >= KRN_BASE && byteaddr < KRN_BASE + 5) begin
            case (byteaddr[2:0])
                3'd0: read_data <= kernel_bits[7:0];
                3'd1: read_data <= kernel_bits[15:8];
                3'd2: read_data <= kernel_bits[23:16];
                3'd3: read_data <= {5'b0, kernel_bits[26:24]};
                default: read_data <= {6'b0, mode_reg};
            endcase
        end
        else if (byteaddr >= OUT_BASE && byteaddr < OUT_BASE + N_OUT_BYTE)
            read_data <= outoff[0] ? out_word[15:8] : out_word[7:0];
        else if (reg_address == SIG_ADDR) begin
            // Fixed design signature. The host asserts this BEFORE capturing, so a board
            // still holding a previous bitstream cannot be mistaken for this design --
            // the failure that voided the 2026-09-06 amplifier sweep. Amplitude cannot
            // separate similar builds; a signature can.
            case (reg_bytecnt)
                7'd0: read_data <= "R";
                7'd1: read_data <= "G";
                7'd2: read_data <= "B";
                7'd3: read_data <= "S";
                7'd4: read_data <= "C";
                7'd5: read_data <= "A";
                7'd6: read_data <= "0";
                7'd7: read_data <= "1";
                default: read_data <= 8'h00;
            endcase
        end
        else if (reg_address == STAT_ADDR) begin
            case (reg_bytecnt)
                7'd0: read_data <= {7'b0, busy_usb};
                7'd1: read_data <= go_count_usb;
                7'd2: read_data <= heartbeat_usb[23:16];
                7'd3: read_data <= start_count_usb;
                7'd4: read_data <= write_count_usb[7:0];
                7'd5: read_data <= write_count_usb[15:8];
                7'd6: read_data <= last_out_addr_usb[7:0];
                7'd7: read_data <= {6'b0, last_out_addr_usb[9:8]};
                7'd8: read_data <= 8'hA5;                       // status-page marker
                7'd9: read_data <= {7'b0, reset_i};
                7'd10: read_data <= reg_address[7:0];
                7'd11: read_data <= {1'b0, reg_bytecnt};
                7'd12: read_data <= image_r[0];                 // per-plane write sanity
                7'd13: read_data <= image_g[0];
                7'd14: read_data <= image_b[0];
                7'd15: read_data <= kernel_bits[7:0];
                7'd16: read_data <= {5'b0, kernel_bits[26:24]};
                7'd17: read_data <= {6'b0, mode_reg};
                7'd18: read_data <= OUT_SIDE[7:0];              // geometry sanity
                7'd19: read_data <= IMG_SIDE[7:0];
                7'd20: read_data <= 8'h5A;                      // end marker
                default: read_data <= 8'h00;
            endcase
        end
        else
            read_data <= 8'h00;
    end

    initial begin
        if ((IMG_SIDE & (IMG_SIDE-1)) != 0)
            $error("IMG_SIDE must be a power of two: the plane decode is a bit slice");
    end

    wire go_crypto;
    cdc_pulse U_go (
        .reset_i(reset_i), .src_clk(usb_clk), .src_pulse(go_pulse_usb),
        .dst_clk(crypto_clk), .dst_pulse(go_crypto)
    );

    (* ASYNC_REG="TRUE" *) reg [2:0] extt;
    always @(posedge crypto_clk) begin
        if (reset_i) extt <= 3'b000;
        else         extt <= {extt[1:0], exttrigger_in};
    end
    wire ext_start = extt[1] & ~extt[2];

    // Mode crosses into the crypto domain. It is only changed while idle, between
    // captures, so a two-flop synchronizer is sufficient and no handshake is needed.
    (* ASYNC_REG="TRUE" *) reg [1:0] mode_meta, mode_crypto;
    always @(posedge crypto_clk) begin
        if (reset_i) begin
            mode_meta   <= 2'd0;
            mode_crypto <= 2'd0;
        end else begin
            mode_meta   <= mode_reg;
            mode_crypto <= mode_meta;
        end
    end

    wire busy_crypto;
    wire core_trigger;
    wire [9:0] core_out_addr;
    wire signed [15:0] core_out_data;
    wire core_out_we;
    wire core_leak;

    // The 27 taps, read at the origin the core publishes. Tap index is c*9 + r*3 + col,
    // matching rgb_leakage_core's window_bytes layout. Row stride is IMG_SIDE.
    wire [9:0] win_base_full;
    wire [PLANE_W-1:0] win_base = win_base_full[PLANE_W-1:0];
    localparam [PLANE_W-1:0] R1 = IMG_SIDE[PLANE_W-1:0];
    localparam [PLANE_W-1:0] R2 = IMG_SIDE[PLANE_W-1:0] * 2;

    wire [215:0] win_bytes = {
        image_b[win_base + R2 + 2], image_b[win_base + R2 + 1],
        image_b[win_base + R2], image_b[win_base + R1 + 2],
        image_b[win_base + R1 + 1], image_b[win_base + R1],
        image_b[win_base + 2],      image_b[win_base + 1],
        image_b[win_base],
        image_g[win_base + R2 + 2], image_g[win_base + R2 + 1],
        image_g[win_base + R2], image_g[win_base + R1 + 2],
        image_g[win_base + R1 + 1], image_g[win_base + R1],
        image_g[win_base + 2],      image_g[win_base + 1],
        image_g[win_base],
        image_r[win_base + R2 + 2], image_r[win_base + R2 + 1],
        image_r[win_base + R2], image_r[win_base + R1 + 2],
        image_r[win_base + R1 + 1], image_r[win_base + R1],
        image_r[win_base + 2],      image_r[win_base + 1],
        image_r[win_base]
    };

    rgb_leakage_core #(.DWELL(DWELL), .IMG_SIDE(IMG_SIDE)) U_core (
        .clk(crypto_clk), .reset(reset_i), .start(go_crypto | ext_start),
        .mode(mode_crypto),
        .window_base(win_base_full), .window_bytes(win_bytes),
        .kernel_bits(kernel_bits),
        .busy(busy_crypto), .trigger(core_trigger),
        .out_addr(core_out_addr), .out_data(core_out_data), .out_we(core_out_we),
        .leak_sink(core_leak)
    );

    wire start_event = go_crypto | ext_start;
    always @(posedge crypto_clk) begin
        if (reset_i) begin
            crypto_heartbeat     <= 24'd0;
            start_count_crypto   <= 8'd0;
            write_count_crypto   <= 16'd0;
            last_out_addr_crypto <= 10'd0;
        end else begin
            crypto_heartbeat <= crypto_heartbeat + 24'd1;
            if (start_event)
                start_count_crypto <= start_count_crypto + 8'd1;
            if (core_out_we) begin
                write_count_crypto   <= write_count_crypto + 16'd1;
                last_out_addr_crypto <= core_out_addr;
            end
        end
    end

    always @(posedge crypto_clk)
        if (core_out_we)
            out_mem[core_out_addr] <= core_out_data;

    (* ASYNC_REG="TRUE" *) reg [1:0] busy_sync;
    always @(posedge usb_clk) begin
        if (reset_i) busy_sync <= 2'b00;
        else         busy_sync <= {busy_sync[0], busy_crypto};
    end
    assign busy_usb   = busy_sync[1];
    assign leak_led   = core_leak;
    assign tio_trigger = core_trigger;
endmodule

`default_nettype wire
