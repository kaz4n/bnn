`timescale 1ns/1ps
`default_nettype none
// Intentional RGB -> RGGB sampling. No physical side-channel sensor.
module cw305_rgb_top(
    input wire usb_clk,
    inout wire [7:0] usb_data,
    input wire [20:0] usb_addr,
    input wire usb_rdn, usb_wrn, usb_cen,
    input wire pushbutton,
    output wire led1, led2, led3
);
    wire clk;
    BUFG usb_clock_buffer(.I(usb_clk), .O(clk));
    reg [4:0] startup = 0;
    (* ASYNC_REG = "TRUE" *) reg [1:0] button_sync = 0;
    always @(posedge clk) begin
        if (!(&startup)) startup <= startup + 1'b1;
        button_sync <= {button_sync[0], pushbutton};
    end
    wire reset = !(&startup) || !button_sync[1];
    wire [4:0] address;
    wire [15:0] offset;
    wire [7:0] write_data, usb_out;
    reg [7:0] read_data;
    wire reg_read, reg_write, isout;
    assign usb_data = isout ? usb_out : 8'bz;
    cw305_usb_reg_fe #(.pADDR_WIDTH(21), .pBYTECNT_SIZE(16)) frontend(
        .usb_clk(clk), .rst(reset), .usb_din(usb_data), .usb_dout(usb_out),
        .usb_isout(isout), .usb_addr(usb_addr), .usb_rdn(usb_rdn),
        .usb_wrn(usb_wrn), .usb_alen(1'b0), .usb_cen(usb_cen),
        .reg_address(address), .reg_bytecnt(offset), .reg_datao(write_data),
        .reg_datai(read_data), .reg_read(reg_read), .reg_write(reg_write),
        .reg_addrvalid()
    );
    // A USB bus strobe can stay asserted for several FPGA clocks. Accept a
    // byte once per address/strobe, rather than duplicating streamed pixels.
    reg last_write;
    reg [20:0] last_address;
    wire write_once = reg_write && (!last_write || last_address != {address,offset});
    reg [3:0] width_log2;
    reg start, soft_reset, pixel_valid;
    reg [7:0] red_byte, green_byte;
    reg [1:0] phase;
    reg [23:0] pixel;
    wire busy, done, error;
    wire [14:0] count;
    wire [7:0] trace_data;
    always @(posedge clk) begin
        start <= 0;
        soft_reset <= 0;
        pixel_valid <= 0;
        if (reset) begin
            last_write <= 0;
            last_address <= 0;
            width_log2 <= 5;
            phase <= 0;
            red_byte <= 0;
            green_byte <= 0;
            pixel <= 0;
        end else begin
            last_write <= reg_write;
            last_address <= {address,offset};
            if (write_once) begin
                case (address)
                    1: if (offset == 0) begin
                        if (write_data[1]) begin soft_reset <= 1; phase <= 0; end
                        else if (write_data[0]) begin start <= 1; phase <= 0; end
                    end
                    2: if (offset == 0) width_log2 <= write_data[3:0];
                    5: begin
                        case (phase)
                            0: begin red_byte <= write_data; phase <= 1; end
                            1: begin green_byte <= write_data; phase <= 2; end
                            2: begin
                                pixel <= {red_byte,green_byte,write_data};
                                pixel_valid <= 1;
                                phase <= 0;
                            end
                            default: phase <= 0;
                        endcase
                    end
                    default: begin end
                endcase
            end
        end
    end
    rgb_bayer_capture capture(
        .clk(clk), .reset(reset || soft_reset), .start(start),
        .width_log2(width_log2), .rgb_valid(pixel_valid), .rgb_pixel(pixel),
        .busy(busy), .done(done), .error(error), .sample_count(count),
        // Sample the external address directly into BRAM on the first clock.
        // Avoid adding the frontend's address register to RAM read latency.
        .trace_read_en(usb_addr[20:16] == 6), .trace_read_addr(usb_addr[13:0]),
        .trace_read_data(trace_data)
    );
    always @* begin
        read_data = 0;
        case (address)
            0: case (offset)
                0: read_data = "R"; 1: read_data = "G";
                2: read_data = "B"; 3: read_data = "B";
                4: read_data = "A"; 5: read_data = "Y";
                6: read_data = "0"; 7: read_data = "1";
                default: read_data = 0;
            endcase
            2: if (offset == 0) read_data = {4'b0,width_log2};
            3: case (offset)
                0: read_data = {5'b0,error,done,busy};
                1: read_data = count[7:0];
                2: read_data = {1'b0,count[14:8]};
                3: read_data = {6'b0,phase};
                default: read_data = 0;
            endcase
            6: read_data = trace_data;
            default: read_data = 0;
        endcase
    end
    assign led1 = busy;
    assign led2 = done;
    assign led3 = error;
endmodule
`default_nettype wire
