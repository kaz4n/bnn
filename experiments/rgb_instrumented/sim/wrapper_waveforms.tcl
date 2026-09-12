# Selected bus and pipeline signals only; do not log frame-memory arrays.
# The VCD/WDB records the one-setup + three-NRD-low read profile, including
# consecutive read transactions and the exact sampling edge markers.
set read_signals [list \
    /tb_cw305_rgb_top/usb_clk \
    /tb_cw305_rgb_top/usb_addr \
    /tb_cw305_rgb_top/usb_rdn \
    /tb_cw305_rgb_top/usb_wrn \
    /tb_cw305_rgb_top/usb_cen \
    /tb_cw305_rgb_top/usb_data \
    /tb_cw305_rgb_top/read_phase \
    /tb_cw305_rgb_top/read_sample_toggle \
    /tb_cw305_rgb_top/sampled_read_address \
    /tb_cw305_rgb_top/sampled_read_data \
    /tb_cw305_rgb_top/dut/address \
    /tb_cw305_rgb_top/dut/offset \
    /tb_cw305_rgb_top/dut/trace_data \
    /tb_cw305_rgb_top/dut/isout]
open_vcd minimum_read_timing.vcd
log_vcd $read_signals
log_wave $read_signals
run all
close_vcd
quit
