# RGB line-buffer leakage build. No HLS, no RO sensor, no pblock, no leakage amplifier.
#
# Uses rgb_linebuf_core + cw305_reg_linebuf_rgb: image in block RAM, streamed one pixel
# per window advance into a line buffer. The earlier random-access variant
# (rgb_leakage_core + cw305_reg_leakage_rgb) is superseded and must not be built -- it
# synthesised to 115% of the part.
# Run from build/: vivado -mode batch -source run_vivado_leakage_rgb.tcl
#
# There is no LEAK_LANES here: this design has no amplifier to size. The channel
# dataflow, which is the experimental variable, is a RUNTIME register rather than a
# build-time parameter, so one bitstream covers all three conditions. That is deliberate
# -- reprogramming the CW305 over a configured FPGA silently fails on this board, and a
# per-condition rebuild is exactly how the 2026-09-06 amplifier sweep captured the same
# bitstream four times (attack/hardtests_20260830/AMP_SWEEP_VOID.md).

set DWELL 8
if {[info exists ::env(DWELL)]} { set DWELL $::env(DWELL) }
set IMG_SIDE 32
if {[info exists ::env(IMG_SIDE)]} { set IMG_SIDE $::env(IMG_SIDE) }
set BUILD_NAME leakage_rgblb_d${DWELL}_s${IMG_SIDE}
set part xc7a100tftg256-2

set CWHDL "C:/Users/narut/ChipWhisperer/chipwhisperer/firmware/fpgas/aes/hdl"
set CWXDC "C:/Users/narut/ChipWhisperer/chipwhisperer/firmware/fpgas/aes/vivado/cw305.xdc"
if {[info exists ::env(CW305_SRC)]} { set CWHDL $::env(CW305_SRC) }

foreach f [list \
    "$CWHDL/cw305_usb_reg_fe.v" "$CWHDL/clocks.v" \
    ../cw305/cdc_pulse.v ../cw305/rgb_linebuf_core.sv \
    ../cw305/cw305_reg_linebuf_rgb.sv ../cw305/cw305_leakage_rgb_top.sv \
    $CWXDC ../cw305/cw305_leakage_rgb.xdc] {
    if {![file exists $f]} { error "Required input missing: $f" }
}

create_project -force cw305_${BUILD_NAME} ./vivado_${BUILD_NAME} -part $part
add_files "$CWHDL/cw305_usb_reg_fe.v"
add_files "$CWHDL/clocks.v"
add_files ../cw305/cdc_pulse.v
add_files -fileset sources_1 ../cw305/rgb_linebuf_core.sv
add_files -fileset sources_1 ../cw305/cw305_reg_linebuf_rgb.sv
add_files -fileset sources_1 ../cw305/cw305_leakage_rgb_top.sv
set_property file_type SystemVerilog [get_files *.sv]

set_property top cw305_leakage_rgb_top [current_fileset]
set_property generic "DWELL=$DWELL IMG_SIDE=$IMG_SIDE" [current_fileset]
add_files -fileset constrs_1 $CWXDC
add_files -fileset constrs_1 ../cw305/cw305_leakage_rgb.xdc
update_compile_order -fileset sources_1

launch_runs synth_1 -jobs 4
wait_on_run synth_1
if {[get_property PROGRESS [get_runs synth_1]] != "100%"} {
    error "Synthesis failed"
}
launch_runs impl_1 -to_step write_bitstream -jobs 4
wait_on_run impl_1
if {[get_property PROGRESS [get_runs impl_1]] != "100%"} {
    error "Implementation failed"
}

open_run impl_1
report_utilization -file ${BUILD_NAME}_utilization.rpt
report_timing_summary -file ${BUILD_NAME}_timing.rpt
report_clock_interaction -file ${BUILD_NAME}_clock_interaction.rpt

# Fail loudly on a negative-slack build rather than shipping a bitstream that meets
# nothing. The grey builds all closed with positive slack; a regression here means the
# RGB store did not fit the way the three-plane decomposition predicts.
set wns [get_property STATS.WNS [get_runs impl_1]]
set whs [get_property STATS.WHS [get_runs impl_1]]
puts "TIMING: WNS=$wns WHS=$whs"
if {$wns < 0 || $whs < 0} {
    error "Timing not met: WNS=$wns WHS=$whs"
}

set bitfile ./vivado_${BUILD_NAME}/cw305_${BUILD_NAME}.runs/impl_1/cw305_leakage_rgb_top.bit
file copy -force $bitfile ./cw305_${BUILD_NAME}.bit
puts "BITSTREAM: build/cw305_${BUILD_NAME}.bit"
exit
