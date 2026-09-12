`timescale 1ns / 1ps
`default_nettype none

// Wrapper RTL simulation with the stock NewAE USB frontend and a BUFG stub.
// USB address/data/control inputs are synchronous to usb_clk. This does not
// model USB electrical behavior or claim a physical board acquisition.
module tb_cw305_rgb_top;
    // Nominal 96 MHz shared MCK/PCK0 profile supplied for this validation.
    // 1 ps time resolution rounds the clock period to 10.416 ns.
    localparam real USB_HALF_PERIOD_NS = 1000.0 / (2.0 * 96.0);
    reg usb_clk = 1'b0;
    always #(USB_HALF_PERIOD_NS) usb_clk = ~usb_clk;
    wire [7:0] usb_data;
    reg [20:0] usb_addr = 21'd0;
    reg usb_rdn = 1'b1;
    reg usb_wrn = 1'b1;
    reg usb_cen = 1'b1;
    reg pushbutton = 1'b1;
    reg host_drive = 1'b0;
    reg [7:0] host_data = 8'd0;
    wire led1, led2, led3;
    assign usb_data = host_drive ? host_data : 8'bz;

    reg [7:0] expected [0:16383];
    integer checked_bytes = 0;
    integer minimum_read_transactions = 0;
    integer consecutive_read_pairs = 0;
    integer signature_index;
    reg [7:0] read_value;
    realtime previous_read_end = -1.0;
    // Waveform markers: 1 = setup, 2 = NRD pulse, 0 = transaction ended.
    reg [1:0] read_phase = 0;
    reg read_sample_toggle = 0;
    reg [7:0] sampled_read_data = 0;
    reg [20:0] sampled_read_address = 0;

    cw305_rgb_top dut (
        .usb_clk(usb_clk), .usb_data(usb_data), .usb_addr(usb_addr),
        .usb_rdn(usb_rdn), .usb_wrn(usb_wrn), .usb_cen(usb_cen),
        .pushbutton(pushbutton), .led1(led1), .led2(led2), .led3(led3)
    );

    task automatic idle_bus(input integer clocks);
        begin
            @(negedge usb_clk);
            usb_rdn = 1'b1;
            usb_wrn = 1'b1;
            usb_cen = 1'b1;
            host_drive = 1'b0;
            repeat (clocks) @(posedge usb_clk);
            #1;
        end
    endtask

    // Can be called back-to-back with the strobe continuously active. Offset
    // increments alone distinguish bytes; each byte may be held 1..5 clocks.
    task automatic write_byte (
        input reg [4:0] register_address,
        input integer byte_offset,
        input reg [7:0] value,
        input integer hold_clocks
    );
        begin
            @(negedge usb_clk);
            usb_addr = {register_address, byte_offset[15:0]};
            usb_rdn = 1'b1;
            usb_wrn = 1'b0;
            usb_cen = 1'b0;
            host_drive = 1'b1;
            host_data = value;
            repeat (hold_clocks) @(posedge usb_clk);
            #1;
        end
    endtask

    task automatic write_register (
        input reg [4:0] register_address,
        input reg [7:0] value
    );
        begin
            idle_bus(5);
            write_byte(register_address, 0, value, 4);
            idle_bus(5);
        end
    endtask

    task automatic read_byte (
        input reg [4:0] register_address,
        input integer byte_offset,
        output reg [7:0] value
    );
        realtime transaction_start;
        realtime pulse_start;
        begin
            // A directly consecutive call begins at the previous sample
            // edge, so adjacent transactions take exactly four clocks each.
            if ($realtime == previous_read_end)
                consecutive_read_pairs = consecutive_read_pairs + 1;
            else
                @(negedge usb_clk);
            transaction_start = $realtime;
            usb_addr = {register_address, byte_offset[15:0]};
            usb_rdn = 1'b1;
            usb_wrn = 1'b1;
            usb_cen = 1'b0;
            host_drive = 1'b0;
            read_phase = 1;
            // NRD_SETUP=1: hold address for one clock before asserting NRD.
            @(negedge usb_clk);
            pulse_start = $realtime;
            usb_rdn = 1'b0;
            read_phase = 2;
            // NRD_PULSE=3, NRD_CYCLE=4: sample immediately at the end of
            // exactly three low-strobe clocks, before deasserting NRD.
            repeat (3) @(negedge usb_clk);
            value = usb_data;
            sampled_read_data = usb_data;
            sampled_read_address = usb_addr;
            read_sample_toggle = !read_sample_toggle;
            assert (($realtime - transaction_start > 41.663) &&
                    ($realtime - transaction_start < 41.665) &&
                    ($realtime - pulse_start > 31.247) &&
                    ($realtime - pulse_start < 31.249))
                else $fatal(1, "Wrapper RTL simulation minimum read timing duration mismatch");
            minimum_read_transactions = minimum_read_transactions + 1;
            previous_read_end = $realtime;
            usb_rdn = 1'b1;
            read_phase = 0;
        end
    endtask

    task automatic check_status (
        input reg [2:0] status,
        input integer count,
        input reg [1:0] phase
    );
        reg [7:0] value;
        begin
            read_byte(5'd3, 0, value);
            assert (value === {5'd0, status})
                else $fatal(1, "Wrapper RTL simulation status: got %02h expected %02h", value, status);
            assert ({led3, led2, led1} === status)
                else $fatal(1, "Wrapper RTL simulation LED/status mismatch");
            read_byte(5'd3, 1, value);
            assert (value === count[7:0])
                else $fatal(1, "Wrapper RTL simulation count low: got %02h expected %02h", value, count[7:0]);
            read_byte(5'd3, 2, value);
            assert (value === {1'b0, count[14:8]})
                else $fatal(1, "Wrapper RTL simulation count high: got %02h expected %02h", value, {1'b0, count[14:8]});
            read_byte(5'd3, 3, value);
            assert (value === {6'd0, phase})
                else $fatal(1, "Wrapper RTL simulation phase: got %02h expected %0d", value, phase);
            idle_bus(5);
        end
    endtask

    function automatic [7:0] signature_byte(input integer index);
        begin
            case (index)
                0: signature_byte = "R";
                1: signature_byte = "G";
                2: signature_byte = "B";
                3: signature_byte = "B";
                4: signature_byte = "A";
                5: signature_byte = "Y";
                6: signature_byte = "0";
                7: signature_byte = "1";
                default: signature_byte = 0;
            endcase
        end
    endfunction

    function automatic [23:0] make_pixel (
        input integer row,
        input integer column,
        input integer frame_tag
    );
        reg [7:0] red_value, green_value, blue_value;
        begin
            red_value = row * 31 + column * 7 + frame_tag * 13 + 9;
            green_value = row * 5 + column * 23 + frame_tag * 17 + 67;
            blue_value = row * 19 + column * 11 + frame_tag * 3 + 159;
            make_pixel = {red_value, green_value, blue_value};
        end
    endfunction

    // Streaming chunks deliberately split RGB triplets and repeatedly reset
    // the USB byte offset to zero. Framing must follow the stream's phase.
    task automatic run_frame(input integer log2_width);
        integer width;
        integer pixels;
        integer stream_index;
        integer chunk_index;
        integer chunk_length;
        integer chunk_offset;
        integer pixel_index;
        integer row;
        integer column;
        integer address;
        reg [23:0] pixel;
        reg [7:0] value;
        reg [1:0] expected_phase;
        begin
            width = 1 << log2_width;
            pixels = width * width;
            write_register(5'd1, 8'd2);
            check_status(3'b000, 0, 0);
            write_register(5'd2, log2_width[7:0]);
            read_byte(5'd2, 0, value);
            assert (value === log2_width[7:0])
                else $fatal(1, "Wrapper RTL simulation width register mismatch");
            idle_bus(5);
            write_register(5'd1, 8'd1);
            check_status(3'b001, 0, 0);

            stream_index = 0;
            chunk_index = 0;
            while (stream_index < pixels * 3) begin
                case (chunk_index % 6)
                    0: chunk_length = 1;
                    1: chunk_length = 1;
                    2: chunk_length = 511;
                    3: chunk_length = 1024;
                    4: chunk_length = 17;
                    default: chunk_length = 256;
                endcase
                if (chunk_length > pixels * 3 - stream_index)
                    chunk_length = pixels * 3 - stream_index;

                for (chunk_offset = 0; chunk_offset < chunk_length; chunk_offset = chunk_offset + 1) begin
                    pixel_index = stream_index / 3;
                    row = pixel_index / width;
                    column = pixel_index % width;
                    pixel = make_pixel(row, column, log2_width);
                    if (((row % 2) == 0) && ((column % 2) == 0))
                        expected[pixel_index] = pixel[23:16];
                    else if (((row % 2) == 1) && ((column % 2) == 1))
                        expected[pixel_index] = pixel[7:0];
                    else
                        expected[pixel_index] = pixel[15:8];
                    case (stream_index % 3)
                        0: value = pixel[23:16];
                        1: value = pixel[15:8];
                        default: value = pixel[7:0];
                    endcase
                    write_byte(5'd5, chunk_offset, value, 1 + ((stream_index * 7) % 5));
                    stream_index = stream_index + 1;
                end
                idle_bus(5);
                expected_phase = stream_index % 3;
                if (stream_index == pixels * 3)
                    check_status(3'b010, pixels, expected_phase);
                else
                    check_status(3'b001, stream_index / 3, expected_phase);
                chunk_index = chunk_index + 1;
            end

            // One complete extra pixel must set error without altering data.
            write_byte(5'd5, 0, 8'hFF, 5);
            write_byte(5'd5, 1, 8'hFF, 5);
            write_byte(5'd5, 2, 8'hFF, 5);
            idle_bus(5);
            check_status(3'b110, pixels, 0);
            for (address = 0; address < pixels; address = address + 1) begin
                read_byte(5'd6, address, value);
                assert (value === expected[address])
                    else $fatal(1,
                        "Wrapper RTL simulation readback %0dx%0d address %0d: got %02h expected %02h",
                        width, width, address, value, expected[address]);
                checked_bytes = checked_bytes + 1;
            end
            idle_bus(5);
            check_status(3'b110, pixels, 0);
            $display("Wrapper RTL simulation PASS: %0dx%0d, %0d RGB bytes, %0d reset-offset chunks, %0d readback bytes", width, width, pixels * 3, chunk_index, pixels);
        end
    endtask

    initial begin
        idle_bus(40);
        for (signature_index = 0; signature_index < 9; signature_index = signature_index + 1) begin
            read_byte(5'd0, signature_index, read_value);
            assert (read_value === signature_byte(signature_index))
                else $fatal(1, "Wrapper RTL simulation signature byte %0d: got %02h", signature_index, read_value);
        end
        idle_bus(5);
        read_byte(5'd2, 0, read_value);
        assert (read_value === 8'd5)
            else $fatal(1, "Wrapper RTL simulation default width must be 5");
        idle_bus(5);
        check_status(3'b000, 0, 0);

        // Width writes at other byte offsets must not change configuration.
        write_byte(5'd2, 1, 8'd7, 4);
        idle_bus(5);
        read_byte(5'd2, 0, read_value);
        assert (read_value === 8'd5)
            else $fatal(1, "Wrapper RTL simulation ignored-offset width write changed configuration");
        idle_bus(5);

        write_register(5'd2, 8'd4);
        write_register(5'd1, 8'd1);
        check_status(3'b100, 0, 0);
        run_frame(5);
        run_frame(7);

        // A control reset clears a partial RGB triplet as well as status.
        write_register(5'd1, 8'd2);
        write_register(5'd2, 8'd5);
        write_register(5'd1, 8'd1);
        write_byte(5'd5, 0, 8'h12, 4);
        idle_bus(5);
        check_status(3'b001, 0, 1);
        write_register(5'd1, 8'd2);
        check_status(3'b000, 0, 0);

        assert (checked_bytes == 17408)
            else $fatal(1, "Wrapper RTL simulation checked byte count mismatch");
        assert (consecutive_read_pairs > 17408)
            else $fatal(1, "Wrapper RTL simulation insufficient consecutive four-cycle reads");
        $display("Wrapper RTL simulation PASS: minimum read profile, %0d transactions, %0d consecutive pairs, setup=1 pulse=3 cycle=4 at nominal 96 MHz", minimum_read_transactions, consecutive_read_pairs);
        $display("Wrapper RTL simulation PASS: all tests, 17408 frame bytes verified. No physical hardware result is implied.");
        $finish;
    end

    initial begin
        #20000000;
        $fatal(1, "Wrapper RTL simulation timeout");
    end
endmodule

`default_nettype wire
