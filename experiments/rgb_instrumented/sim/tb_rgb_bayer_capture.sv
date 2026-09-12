`timescale 1ns / 1ps
`default_nettype none

// This testbench reports RTL simulation results, not physical measurements.
module tb_rgb_bayer_capture;
    reg clk = 1'b0;
    always #5 clk = ~clk;

    reg reset = 1'b1;
    reg start = 1'b0;
    reg [3:0] width_log2 = 4'd5;
    reg rgb_valid = 1'b0;
    reg [23:0] rgb_pixel = 24'd0;
    wire busy;
    wire done;
    wire error;
    wire [14:0] sample_count;
    reg trace_read_en = 1'b0;
    reg [13:0] trace_read_addr = 14'd0;
    wire [7:0] trace_read_data;

    reg [7:0] expected [0:16383];
    reg [31:0] gap_state = 32'hC001D00D;
    integer checked_bytes = 0;
    integer width_case;

    rgb_bayer_capture dut (
        .clk(clk), .reset(reset), .start(start), .width_log2(width_log2),
        .rgb_valid(rgb_valid), .rgb_pixel(rgb_pixel), .busy(busy),
        .done(done), .error(error), .sample_count(sample_count),
        .trace_read_en(trace_read_en), .trace_read_addr(trace_read_addr),
        .trace_read_data(trace_read_data)
    );

    task automatic cycle (
        input reg next_start,
        input reg next_valid,
        input reg [3:0] next_width,
        input reg [23:0] next_pixel
    );
        begin
            @(negedge clk);
            start = next_start;
            rgb_valid = next_valid;
            width_log2 = next_width;
            rgb_pixel = next_pixel;
            @(posedge clk);
            #1;
        end
    endtask

    task automatic check_state (
        input reg expected_busy,
        input reg expected_done,
        input reg expected_error,
        input integer expected_count
    );
        begin
            assert ((busy === expected_busy) &&
                    (done === expected_done) &&
                    (error === expected_error) &&
                    (sample_count === expected_count[14:0]))
                else $fatal(1,
                    "RTL simulation state mismatch at %0t: busy/done/error/count=%b/%b/%b/%0d expected=%b/%b/%b/%0d",
                    $time, busy, done, error, sample_count,
                    expected_busy, expected_done, expected_error, expected_count);
        end
    endtask

    // Distinct arithmetic for each input channel, including row dependence,
    // catches channel swaps, transposed images, wrong widths, and stale data.
    function automatic [23:0] make_pixel (
        input integer row,
        input integer column,
        input integer frame_tag
    );
        reg [7:0] red_value;
        reg [7:0] green_value;
        reg [7:0] blue_value;
        begin
            red_value = row * 17 + column * 3 + frame_tag * 29 + 8;
            green_value = row * 7 + column * 19 + frame_tag * 11 + 83;
            blue_value = row * 23 + column * 5 + frame_tag * 37 + 161;
            make_pixel = {red_value, green_value, blue_value};
        end
    endfunction

    // The reference walks rows/columns explicitly, independently of the
    // DUT's count-bit implementation of Bayer coordinates.
    function automatic [7:0] reference_sample (
        input integer row,
        input integer column,
        input reg [23:0] pixel
    );
        begin
            if (((row % 2) == 0) && ((column % 2) == 0))
                reference_sample = pixel[23:16];
            else if (((row % 2) == 1) && ((column % 2) == 1))
                reference_sample = pixel[7:0];
            else
                reference_sample = pixel[15:8];
        end
    endfunction

    task automatic read_and_check (input integer address);
        reg [7:0] previous_data;
        begin
            previous_data = trace_read_data;
            @(negedge clk);
            trace_read_en = 1'b1;
            trace_read_addr = address[13:0];
            #1;
            assert (trace_read_data === previous_data)
                else $fatal(1, "RTL simulation: read data changed before a rising edge");
            @(posedge clk);
            #1;
            assert (trace_read_data === expected[address])
                else $fatal(1,
                    "RTL simulation byte mismatch at address %0d: got %02h expected %02h",
                    address, trace_read_data, expected[address]);
            checked_bytes = checked_bytes + 1;
        end
    endtask

    task automatic run_frame (input integer log2_width);
        integer width;
        integer count;
        integer row;
        integer column;
        integer gaps;
        integer gap_index;
        integer address;
        reg expected_error;
        reg [23:0] pixel;
        reg [7:0] held_data;
        begin
            width = 1 << log2_width;
            count = 0;
            expected_error = 1'b0;

            // Start takes priority even if a pixel is offered on that edge.
            cycle(1'b1, 1'b1, log2_width[3:0], 24'hFEDCBA);
            check_state(1'b1, 1'b0, 1'b0, 0);
            cycle(1'b0, 1'b0, 4'd0, 24'd0);
            check_state(1'b1, 1'b0, 1'b0, 0);

            for (row = 0; row < width; row = row + 1) begin
                for (column = 0; column < width; column = column + 1) begin
                    // Deterministic pseudo-random gaps of 0..3 cycles.
                    gap_state = (gap_state * 32'd1664525) + 32'd1013904223;
                    gaps = {30'd0, gap_state[31:30]};
                    for (gap_index = 0; gap_index < gaps; gap_index = gap_index + 1) begin
                        cycle(1'b0, 1'b0, 4'd0, 24'hFFFFFF);
                        check_state(1'b1, 1'b0, expected_error, count);
                    end

                    if (count == width + 2) begin
                        // A valid-width restart while busy is rejected. The
                        // simultaneous sentinel pixel must also be ignored.
                        cycle(1'b1, 1'b1, 4'd7, 24'h123456);
                        expected_error = 1'b1;
                        check_state(1'b1, 1'b0, expected_error, count);
                    end

                    pixel = make_pixel(row, column, log2_width);
                    expected[count] = reference_sample(row, column, pixel);
                    // Change width while busy to prove the configured width
                    // was latched, including over every row boundary.
                    cycle(1'b0, 1'b1, 4'd0, pixel);
                    count = count + 1;
                    if (count == width * width)
                        check_state(1'b0, 1'b1, expected_error, count);
                    else
                        check_state(1'b1, 1'b0, expected_error, count);
                end
            end

            // Completion is sticky; extra pixels cannot wrap and overwrite
            // the first byte or any other completed-frame storage.
            cycle(1'b0, 1'b1, 4'd5, 24'hFFFFFF);
            check_state(1'b0, 1'b1, 1'b1, count);
            cycle(1'b0, 1'b1, 4'd5, 24'h000000);
            check_state(1'b0, 1'b1, 1'b1, count);
            cycle(1'b0, 1'b0, 4'd5, 24'd0);
            check_state(1'b0, 1'b1, 1'b1, count);

            for (address = 0; address < count; address = address + 1)
                read_and_check(address);

            // Read enable is a real enable: changing the address alone must
            // not change the registered read output.
            held_data = trace_read_data;
            @(negedge clk);
            trace_read_en = 1'b0;
            trace_read_addr = 14'd0;
            @(posedge clk);
            #1;
            assert (trace_read_data === held_data)
                else $fatal(1, "RTL simulation: disabled read did not hold its value");
            check_state(1'b0, 1'b1, 1'b1, count);

            // An unsupported start clears done but preserves completed data
            // and its count. A later valid start clears this error.
            cycle(1'b1, 1'b0, 4'd4, 24'd0);
            check_state(1'b0, 1'b0, 1'b1, count);
            cycle(1'b0, 1'b0, 4'd4, 24'd0);
            check_state(1'b0, 1'b0, 1'b1, count);
            $display("RTL simulation PASS: %0dx%0d, %0d bytes checked, gaps and protocol cases", width, width, count);
        end
    endtask

    initial begin
        repeat (3) @(posedge clk);
        #1;
        check_state(1'b0, 1'b0, 1'b0, 0);
        assert (trace_read_data === 8'd0)
            else $fatal(1, "RTL simulation: synchronous read output reset failed");
        @(negedge clk);
        reset = 1'b0;

        // An unsolicited idle pixel sets an error without producing data.
        cycle(1'b0, 1'b1, 4'd5, 24'h123456);
        check_state(1'b0, 1'b0, 1'b1, 0);
        cycle(1'b0, 1'b0, 4'd5, 24'd0);
        check_state(1'b0, 1'b0, 1'b1, 0);

        for (width_case = 0; width_case < 16; width_case = width_case + 1) begin
            if ((width_case != 5) && (width_case != 6) && (width_case != 7)) begin
                cycle(1'b1, 1'b0, width_case[3:0], 24'd0);
                check_state(1'b0, 1'b0, 1'b1, 0);
                cycle(1'b0, 1'b0, width_case[3:0], 24'd0);
                check_state(1'b0, 1'b0, 1'b1, 0);
            end
        end

        run_frame(5);
        run_frame(6);
        run_frame(7);

        // Synchronous reset aborts a partial frame and clears all status.
        cycle(1'b1, 1'b0, 4'd5, 24'd0);
        check_state(1'b1, 1'b0, 1'b0, 0);
        cycle(1'b0, 1'b1, 4'd5, 24'h123456);
        check_state(1'b1, 1'b0, 1'b0, 1);
        @(negedge clk);
        reset = 1'b1;
        @(posedge clk);
        #1;
        check_state(1'b0, 1'b0, 1'b0, 0);
        @(negedge clk);
        reset = 1'b0;
        rgb_valid = 1'b0;
        cycle(1'b0, 1'b0, 4'd5, 24'd0);
        check_state(1'b0, 1'b0, 1'b0, 0);

        assert (checked_bytes == 21504)
            else $fatal(1, "RTL simulation: unexpected total checked byte count %0d", checked_bytes);
        $display("RTL simulation PASS: all tests, %0d frame bytes verified. No physical hardware result is implied.", checked_bytes);
        $finish;
    end

    initial begin
        #10000000;
        $fatal(1, "RTL simulation timeout");
    end
endmodule

`default_nettype wire
