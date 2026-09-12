# Clean leakage-first MNIST build.  No HLS, RO sensor, pblock, or stale RTL fallback.
# Run from build/: vivado -mode batch -source run_vivado_leakage.tcl

set DWELL 8
if {[info exists ::env(DWELL)]} { set DWELL $::env(DWELL) }
set LEAK_LANES 63
if {[info exists ::env(LEAK_LANES)]} { set LEAK_LANES $::env(LEAK_LANES) }
set BUILD_NAME leakage_wide_d${DWELL}_l${LEAK_LANES}
set part xc7a100tftg256-2

set CWHDL "C:/Users/narut/ChipWhisperer/chipwhisperer/firmware/fpgas/aes/hdl"
set CWXDC "C:/Users/narut/ChipWhisperer/chipwhisperer/firmware/fpgas/aes/vivado/cw305.xdc"
if {[info exists ::env(CW305_SRC)]} { set CWHDL $::env(CW305_SRC) }

foreach f [list \
    "$CWHDL/cw305_usb_reg_fe.v" "$CWHDL/clocks.v" \
    ../cw305/cdc_pulse.v ../cw305/mnist_leakage_core_wide.sv \
    ../cw305/cw305_reg_leakage_wide.sv ../cw305/cw305_leakage_wide_top.sv \
    $CWXDC ../cw305/cw305_leakage_grey.xdc] {
    if {![file exists $f]} { error "Required input missing: $f" }
}

create_project -force cw305_${BUILD_NAME} ./vivado_${BUILD_NAME} -part $part
add_files "$CWHDL/cw305_usb_reg_fe.v"
add_files "$CWHDL/clocks.v"
add_files ../cw305/cdc_pulse.v
add_files -fileset sources_1 ../cw305/mnist_leakage_core_wide.sv
add_files -fileset sources_1 ../cw305/cw305_reg_leakage_wide.sv
add_files -fileset sources_1 ../cw305/cw305_leakage_wide_top.sv
set_property file_type SystemVerilog [get_files *.sv]

set_property top cw305_leakage_wide_top [current_fileset]
set_property generic "DWELL=$DWELL LEAK_LANES=$LEAK_LANES" [current_fileset]
add_files -fileset constrs_1 $CWXDC
add_files -fileset constrs_1 ../cw305/cw305_leakage_grey.xdc
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
set bitfile ./vivado_${BUILD_NAME}/cw305_${BUILD_NAME}.runs/impl_1/cw305_leakage_wide_top.bit
file copy -force $bitfile ./cw305_${BUILD_NAME}.bit
puts "BITSTREAM: build/cw305_${BUILD_NAME}.bit"
exit
