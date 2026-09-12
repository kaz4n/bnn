`timescale 1ns/1ps
`default_nettype none

module cw305_reg_leakage_wide #(
    parameter integer pADDR_WIDTH   = 21,
    parameter integer pBYTECNT_SIZE = 7,
    parameter integer DWELL         = 8,
    parameter integer LEAK_LANES    = 63
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
    localparam integer N_IMAGE  = 784;
    localparam integer N_OUTPUT = 676;
    localparam [pADDR_WIDTH-1:0] OUT_BASE = 21'd2048;
    localparam [pADDR_WIDTH-1:0] KRN_BASE = 21'd4096;
    localparam integer GO_ADDR   = 33;
    localparam integer STAT_ADDR = 34;
    // Debug is multiplexed onto STATUS byte offsets instead of separate high
    // register pages.  Some CW305 setups only exercise low register pages
    // reliably; keeping byte 0 as the busy flag preserves existing host code:
    //   fpga_read(34, 1)[0] -> busy
    //   fpga_read(34, 8)    -> busy, go_count, heartbeat, start_count, ...

    wire [pADDR_WIDTH-1:0] byteaddr = {reg_address, reg_bytecnt};
    wire [pADDR_WIDTH-1:0] outoff = byteaddr - OUT_BASE;

    // Static while engine runs.  Single-bit storage keeps unrelated switching low.
    reg [N_IMAGE-1:0] image_bits;
    reg [8:0] kernel_bits;
    reg signed [7:0] out_mem [0:N_OUTPUT-1];
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

    wire busy_usb;
    always @(posedge usb_clk) begin
        go_pulse_usb <= 1'b0;
        if (reset_i) begin
            O_user_led <= 1'b0;
            image_bits <= {N_IMAGE{1'b0}};
            kernel_bits <= 9'b0;
            go_count_usb <= 8'd0;
        end else if (reg_addrvalid && reg_write) begin
            if (byteaddr < N_IMAGE)
                image_bits[byteaddr[9:0]] <= write_data[0];
            else if (byteaddr == KRN_BASE)
                kernel_bits[7:0] <= write_data;
            else if (byteaddr == KRN_BASE + 1)
                kernel_bits[8] <= write_data[0];
            else if (reg_address == GO_ADDR) begin
                go_pulse_usb <= 1'b1;
                go_count_usb <= go_count_usb + 8'd1;
            end
            else if (reg_address == 14'h01)
                O_user_led <= write_data[0];
        end

        heartbeat_usb <= crypto_heartbeat;
        start_count_usb <= start_count_crypto;
        write_count_usb <= write_count_crypto;
        last_out_addr_usb <= last_out_addr_crypto;

        if (byteaddr < N_IMAGE)
            read_data <= {7'b0, image_bits[byteaddr[9:0]]};
        else if (byteaddr >= KRN_BASE && byteaddr < KRN_BASE + 2)
            read_data <= (byteaddr[0] == 1'b0) ? kernel_bits[7:0] : {7'b0, kernel_bits[8]};
        else if (byteaddr >= OUT_BASE && byteaddr < OUT_BASE + N_OUTPUT)
            read_data <= out_mem[outoff[9:0]];
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
                7'd8: read_data <= 8'hA5;                    // status-page marker
                7'd9: read_data <= {7'b0, reset_i};           // reset sanity
                7'd10: read_data <= reg_address[7:0];         // address sanity
                7'd11: read_data <= {1'b0, reg_bytecnt};      // byte-count sanity
                7'd12: read_data <= image_bits[7:0];          // image write sanity
                7'd13: read_data <= kernel_bits[7:0];         // kernel write sanity
                7'd14: read_data <= {7'b0, kernel_bits[8]};   // kernel high bit
                7'd15: read_data <= 8'h5A;                    // end marker
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
        if (reset_i)
            extt <= 3'b000;
        else
            extt <= {extt[1:0], exttrigger_in};
    end
    wire ext_start = extt[1] & ~extt[2];

    wire busy_crypto;
    wire core_trigger;
    wire [9:0] core_out_addr;
    wire signed [7:0] core_out_data;
    wire core_out_we;
    wire core_leak;

    mnist_leakage_core_wide #(.DWELL(DWELL), .LEAK_LANES(LEAK_LANES)) U_core (
        .clk(crypto_clk), .reset(reset_i), .start(go_crypto | ext_start),
        .image_bits(image_bits), .kernel_bits(kernel_bits),
        .busy(busy_crypto), .trigger(core_trigger),
        .out_addr(core_out_addr), .out_data(core_out_data), .out_we(core_out_we),
        .leak_sink(core_leak)
    );

    wire start_event = go_crypto | ext_start;
    always @(posedge crypto_clk) begin
        if (reset_i) begin
            crypto_heartbeat <= 24'd0;
            start_count_crypto <= 8'd0;
            write_count_crypto <= 16'd0;
            last_out_addr_crypto <= 10'd0;
        end else begin
            crypto_heartbeat <= crypto_heartbeat + 24'd1;
            if (start_event)
                start_count_crypto <= start_count_crypto + 8'd1;
            if (core_out_we) begin
                write_count_crypto <= write_count_crypto + 16'd1;
                last_out_addr_crypto <= core_out_addr;
            end
        end
    end

    always @(posedge crypto_clk)
        if (core_out_we)
            out_mem[core_out_addr] <= core_out_data;

    (* ASYNC_REG="TRUE" *) reg [1:0] busy_sync;
    always @(posedge usb_clk) begin
        if (reset_i)
            busy_sync <= 2'b00;
        else
            busy_sync <= {busy_sync[0], busy_crypto};
    end
    assign busy_usb = busy_sync[1];
    assign leak_led = core_leak;
    assign tio_trigger = core_trigger;
endmodule

`default_nettype wire
