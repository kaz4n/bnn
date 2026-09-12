`timescale 1ns/1ps
`default_nettype none

// RGB register file for the line-buffer leakage core.
//
// SUPERSEDES cw305_reg_leakage_rgb.sv, which stored the image as three distributed-RAM
// planes and fed 27 taps to rgb_leakage_core. Out-of-context synthesis of that pair
// returned 73,174 LUTs -- 115% of an xc7a100t -- with LUT-as-Memory at 1.7%: Vivado
// inferred no distributed RAM and built 27 wide multiplexers out of logic instead.
//
// Here the image lives in a true dual-port block RAM and is STREAMED to the core one
// pixel per window advance. The core owns the line buffer. Block RAM is the natural home
// for 4 KB and the part has 135 tiles of which the previous design used none.
//
// PIXEL WORD: 4 BYTES, NOT 3
// --------------------------
// The store is 1024 words of 32 bits -- {x, B, G, R} -- and the host writes four bytes
// per pixel with the fourth ignored. Three bytes per pixel would make the position
// decode `byteaddr / 3`, which needs a divider or a write pointer that random-access
// writes would break. At four bytes the decode is free wiring: position is
// byteaddr[11:2] and the channel is byteaddr[1:0]. The cost is 1 KB of unused BRAM out
// of a tile that is otherwise idle.
//
// The dataflow mode, and now also the trigger segment window, are runtime registers, so
// one bitstream covers every experimental condition. Reprogramming the CW305 over a
// configured FPGA silently fails on this board and that is exactly how the 2026-09-06
// amplifier sweep captured one bitstream four times.
module cw305_reg_linebuf_rgb #(
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
    localparam integer PLANE      = IMG_SIDE * IMG_SIDE;      // 1024 pixels
    localparam integer IMG_BYTES  = PLANE * 4;                // 4096 bytes (RGBx)
    localparam integer OUT_SIDE   = IMG_SIDE - 2;             // 30
    localparam integer N_OUTPUT   = OUT_SIDE * OUT_SIDE;      // 900
    localparam integer N_OUT_BYTE = N_OUTPUT * 2;             // 1800
    localparam integer PIX_W      = $clog2(PLANE);            // 10

    // Memory map. GO/STAT/SIG decode as byteaddr >> 7, so data pages must avoid their
    // 128-byte windows:
    //   image  0     .. 4095    reg_address  0.. 31
    //   GO/STAT/SIG            reg_address 33/34/35
    //   out    8192  .. 9991    reg_address 64.. 78
    //   ctrl   16384 .. 16392   reg_address 128
    localparam [pADDR_WIDTH-1:0] OUT_BASE  = 21'd8192;
    localparam [pADDR_WIDTH-1:0] CTRL_BASE = 21'd16384;
    localparam integer GO_ADDR   = 33;
    localparam integer STAT_ADDR = 34;
    localparam integer SIG_ADDR  = 35;

    wire [pADDR_WIDTH-1:0] byteaddr = {reg_address, reg_bytecnt};
    wire [pADDR_WIDTH-1:0] outoff   = byteaddr - OUT_BASE;

    wire in_image = (byteaddr < IMG_BYTES);
    wire [PIX_W-1:0] img_pix  = byteaddr[PIX_W+1:2];
    wire [1:0]       img_byte = byteaddr[1:0];

    // True dual-port pixel store: written on usb_clk, read on crypto_clk.
    (* ram_style = "block" *) reg [31:0] image_mem [0:PLANE-1];
    // Readback port. Block RAM adds an output register, so the image readback path is
    // TWO usb_clk cycles deep (BRAM output, then read_data) where the grey design's
    // distributed RAM was one. The CW305 USB front-end holds reg_address stable for many
    // usb_clk cycles during a control transfer, so this is fine on hardware -- but a
    // latency-free image checksum is exposed on the status page as well, so a write can
    // be verified without depending on the readback path at all.
    reg [31:0] img_rd_usb;                       // readback port (usb side)
    reg [15:0] img_checksum;                     // running sum of written image bytes
    reg [23:0] pix_data;                         // streaming port (crypto side)

    reg [26:0] kernel_bits;
    reg [1:0]  mode_reg;
    reg [15:0] seg_start_reg, seg_len_reg;
    reg signed [15:0] out_mem [0:N_OUTPUT-1];
    reg go_pulse_usb;
    reg [7:0] go_count_usb;

    reg [23:0] crypto_heartbeat;
    reg [7:0] start_count_crypto;
    reg [15:0] write_count_crypto;
    reg [15:0] last_out_addr_crypto;
    (* ASYNC_REG="TRUE" *) reg [23:0] heartbeat_usb;
    (* ASYNC_REG="TRUE" *) reg [7:0] start_count_usb;
    (* ASYNC_REG="TRUE" *) reg [15:0] write_count_usb;
    (* ASYNC_REG="TRUE" *) reg [15:0] last_out_addr_usb;

    wire [9:0] out_index = outoff[10:1];
    wire signed [15:0] out_word = out_mem[out_index];

    wire [7:0] img_read = (img_byte == 2'd0) ? img_rd_usb[7:0] :
                          (img_byte == 2'd1) ? img_rd_usb[15:8] :
                          (img_byte == 2'd2) ? img_rd_usb[23:16] : 8'h00;

    wire busy_usb;
    always @(posedge usb_clk) begin
        go_pulse_usb <= 1'b0;
        if (reset_i) begin
            O_user_led    <= 1'b0;
            kernel_bits   <= 27'b0;
            mode_reg      <= 2'd0;
            seg_start_reg <= 16'd0;
            seg_len_reg   <= 16'd0;
            go_count_usb  <= 8'd0;
            img_checksum  <= 16'd0;
        end else if (reg_addrvalid && reg_write) begin
            if (in_image) begin
                case (img_byte)
                    2'd0: image_mem[img_pix][7:0]   <= write_data;
                    2'd1: image_mem[img_pix][15:8]  <= write_data;
                    2'd2: image_mem[img_pix][23:16] <= write_data;
                    default: ;                       // 4th byte is padding
                endcase
                if (img_byte != 2'd3)
                    img_checksum <= img_checksum + {8'd0, write_data};
            end
            else if (byteaddr == CTRL_BASE + 0) kernel_bits[7:0]     <= write_data;
            else if (byteaddr == CTRL_BASE + 1) kernel_bits[15:8]    <= write_data;
            else if (byteaddr == CTRL_BASE + 2) kernel_bits[23:16]   <= write_data;
            else if (byteaddr == CTRL_BASE + 3) kernel_bits[26:24]   <= write_data[2:0];
            else if (byteaddr == CTRL_BASE + 4) mode_reg             <= write_data[1:0];
            else if (byteaddr == CTRL_BASE + 5) seg_start_reg[7:0]   <= write_data;
            else if (byteaddr == CTRL_BASE + 6) seg_start_reg[15:8]  <= write_data;
            else if (byteaddr == CTRL_BASE + 7) seg_len_reg[7:0]     <= write_data;
            else if (byteaddr == CTRL_BASE + 8) seg_len_reg[15:8]    <= write_data;
            else if (reg_address == GO_ADDR) begin
                go_pulse_usb <= 1'b1;
                go_count_usb <= go_count_usb + 8'd1;
            end
            else if (reg_address == 14'h01)
                O_user_led <= write_data[0];
        end

        img_rd_usb        <= image_mem[img_pix];       // registered readback port
        heartbeat_usb     <= crypto_heartbeat;
        start_count_usb   <= start_count_crypto;
        write_count_usb   <= write_count_crypto;
        last_out_addr_usb <= last_out_addr_crypto;

        if (in_image)
            read_data <= img_read;
        else if (byteaddr >= CTRL_BASE && byteaddr < CTRL_BASE + 9) begin
            case (byteaddr[3:0])
                4'd0: read_data <= kernel_bits[7:0];
                4'd1: read_data <= kernel_bits[15:8];
                4'd2: read_data <= kernel_bits[23:16];
                4'd3: read_data <= {5'b0, kernel_bits[26:24]};
                4'd4: read_data <= {6'b0, mode_reg};
                4'd5: read_data <= seg_start_reg[7:0];
                4'd6: read_data <= seg_start_reg[15:8];
                4'd7: read_data <= seg_len_reg[7:0];
                default: read_data <= seg_len_reg[15:8];
            endcase
        end
        else if (byteaddr >= OUT_BASE && byteaddr < OUT_BASE + N_OUT_BYTE)
            read_data <= outoff[0] ? out_word[15:8] : out_word[7:0];
        else if (reg_address == SIG_ADDR) begin
            // Positive design identification. Proving the board is merely "not stock AES"
            // is weaker than it looks -- any stale custom bitstream passes that too.
            case (reg_bytecnt)
                7'd0: read_data <= "R";
                7'd1: read_data <= "G";
                7'd2: read_data <= "B";
                7'd3: read_data <= "L";
                7'd4: read_data <= "B";
                7'd5: read_data <= "U";
                7'd6: read_data <= "F";
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
                7'd7: read_data <= last_out_addr_usb[15:8];
                7'd8: read_data <= 8'hA5;
                7'd9: read_data <= {7'b0, reset_i};
                7'd10: read_data <= reg_address[7:0];
                7'd11: read_data <= {1'b0, reg_bytecnt};
                7'd12: read_data <= kernel_bits[7:0];
                7'd13: read_data <= {6'b0, mode_reg};
                7'd14: read_data <= OUT_SIDE[7:0];
                7'd15: read_data <= IMG_SIDE[7:0];
                7'd16: read_data <= DWELL[7:0];
                7'd17: read_data <= img_checksum[7:0];      // latency-free write sanity
                7'd18: read_data <= img_checksum[15:8];
                7'd19: read_data <= 8'h5A;
                default: read_data <= 8'h00;
            endcase
        end
        else
            read_data <= 8'h00;
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

    // Mode and the segment window are only changed while idle, between captures, so a
    // two-flop synchronizer is sufficient and no handshake is needed.
    (* ASYNC_REG="TRUE" *) reg [1:0]  mode_meta, mode_crypto;
    (* ASYNC_REG="TRUE" *) reg [15:0] segs_meta, segs_crypto, segl_meta, segl_crypto;
    always @(posedge crypto_clk) begin
        if (reset_i) begin
            mode_meta <= 2'd0;  mode_crypto <= 2'd0;
            segs_meta <= 16'd0; segs_crypto <= 16'd0;
            segl_meta <= 16'd0; segl_crypto <= 16'd0;
        end else begin
            mode_meta <= mode_reg;      mode_crypto <= mode_meta;
            segs_meta <= seg_start_reg; segs_crypto <= segs_meta;
            segl_meta <= seg_len_reg;   segl_crypto <= segl_meta;
        end
    end

    wire busy_crypto;
    wire core_trigger;
    wire [15:0] core_out_addr;
    wire signed [15:0] core_out_data;
    wire core_out_we;
    wire core_leak;
    wire [15:0] pix_addr;

    // Streaming read port. Registered, giving the one-cycle latency the core expects.
    always @(posedge crypto_clk)
        pix_data <= image_mem[pix_addr[PIX_W-1:0]][23:0];

    rgb_linebuf_core #(.DWELL(DWELL), .IMG_SIDE(IMG_SIDE)) U_core (
        .clk(crypto_clk), .reset(reset_i), .start(go_crypto | ext_start),
        .mode(mode_crypto), .kernel_bits(kernel_bits),
        .seg_start(segs_crypto), .seg_len(segl_crypto),
        .pix_addr(pix_addr), .pix_data(pix_data),
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
            last_out_addr_crypto <= 16'd0;
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
            out_mem[core_out_addr[9:0]] <= core_out_data;

    (* ASYNC_REG="TRUE" *) reg [1:0] busy_sync;
    always @(posedge usb_clk) begin
        if (reset_i) busy_sync <= 2'b00;
        else         busy_sync <= {busy_sync[0], busy_crypto};
    end
    assign busy_usb    = busy_sync[1];
    assign leak_led    = core_leak;
    assign tio_trigger = core_trigger;
endmodule

`default_nettype wire
