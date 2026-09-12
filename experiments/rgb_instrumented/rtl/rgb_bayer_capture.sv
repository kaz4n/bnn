`timescale 1ns / 1ps
`default_nettype none

// Row-major digital RGGB sampling: R G / G B. All inputs share clk.
// One byte is accepted on each busy && rgb_valid cycle, except that start
// takes priority. Supported square sizes are 32, 64, and 128 pixels.
module rgb_bayer_capture (
    input  wire        clk,
    input  wire        reset,
    input  wire        start,
    input  wire [3:0]  width_log2,
    input  wire        rgb_valid,
    input  wire [23:0] rgb_pixel,
    output reg         busy,
    output reg         done,
    output reg         error,
    output reg  [14:0] sample_count,
    input  wire        trace_read_en,
    input  wire [13:0] trace_read_addr,
    output reg  [7:0]  trace_read_data
);
    reg [3:0] active_width_log2;
    reg [14:0] frame_size;
    reg [7:0] sampled_byte;

    // Storage is deliberately not reset. Only addresses below sample_count
    // belong to the current frame. Read after capture for a complete frame.
    (* ram_style = "block" *) reg [7:0] trace_memory [0:16383];

    // For a power-of-two row width, the count's low bit is column parity
    // and bit width_log2 is row parity. The width is latched at start.
    always @* begin
        case ({sample_count[active_width_log2], sample_count[0]})
            2'b00: sampled_byte = rgb_pixel[23:16];
            2'b11: sampled_byte = rgb_pixel[7:0];
            default: sampled_byte = rgb_pixel[15:8];
        endcase
    end

    // Simple dual-port synchronous RAM. Simultaneous read and write of the
    // same address is outside the interface contract; read completed samples.
    always @(posedge clk) begin
        if (!reset && !start && busy && rgb_valid)
            trace_memory[sample_count[13:0]] <= sampled_byte;
        if (reset)
            trace_read_data <= 8'd0;
        else if (trace_read_en)
            trace_read_data <= trace_memory[trace_read_addr];
    end

    always @(posedge clk) begin
        if (reset) begin
            busy <= 1'b0;
            done <= 1'b0;
            error <= 1'b0;
            sample_count <= 15'd0;
            active_width_log2 <= 4'd5;
            frame_size <= 15'd1024;
        end else if (start) begin
            done <= 1'b0;
            if (busy) begin
                // Reject a restart without losing the partially filled frame.
                error <= 1'b1;
            end else if ((width_log2 == 4'd5) ||
                         (width_log2 == 4'd6) ||
                         (width_log2 == 4'd7)) begin
                busy <= 1'b1;
                error <= 1'b0;
                sample_count <= 15'd0;
                active_width_log2 <= width_log2;
                case (width_log2)
                    4'd5: frame_size <= 15'd1024;
                    4'd6: frame_size <= 15'd4096;
                    default: frame_size <= 15'd16384;
                endcase
            end else begin
                // Invalid starts preserve the last frame and its count.
                error <= 1'b1;
            end
        end else if (rgb_valid) begin
            if (busy) begin
                sample_count <= sample_count + 15'd1;
                if (sample_count == frame_size - 15'd1) begin
                    busy <= 1'b0;
                    done <= 1'b1;
                end
            end else begin
                error <= 1'b1;
            end
        end
    end
endmodule

`default_nettype wire
