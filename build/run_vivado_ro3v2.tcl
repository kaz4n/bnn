# Rebuild the RO bitstream with the v2 oscillation-counting sensor + sensor/conv
# co-location pblock. Produces build/cw305_bnn_ro3v2.bit.
#   set KSIZE=3 ; set FANOUT=16 ; vivado -mode batch -source run_vivado_ro3v2.tcl
# Same flow as run_vivado.tcl; only the sensor file (v2) and the pblock PRE-place hook differ.

set K 3
if {[info exists ::env(KSIZE)]} { set K $::env(KSIZE) }
set BUILD_NAME ro3pb
if {[info exists ::env(BUILD_NAME)]} { set BUILD_NAME $::env(BUILD_NAME) }
set FANOUT 16
if {[info exists ::env(FANOUT)]} { set FANOUT $::env(FANOUT) }
set HLS_NAME bnn_conv1_hls_${K}_f${FANOUT}
set HLS_RTL_DIR ${HLS_NAME}/sol1/impl/verilog
if {![file isdirectory $HLS_RTL_DIR]} { error "Missing HLS RTL dir '$HLS_RTL_DIR'" }
set part xc7a100tftg256-2

set CWHDL "C:/Users/narut/ChipWhisperer/chipwhisperer/firmware/fpgas/aes/hdl"
set CWXDC "C:/Users/narut/ChipWhisperer/chipwhisperer/firmware/fpgas/aes/vivado/cw305.xdc"
if {[info exists ::env(CW305_SRC)]} { set CWHDL $::env(CW305_SRC) }

create_project -force cw305_bnn_${BUILD_NAME} ./vivado_${BUILD_NAME} -part $part
add_files "$CWHDL/cw305_usb_reg_fe.v"
add_files "$CWHDL/clocks.v"
add_files ../cw305/cdc_pulse.v
add_files ../cw305/ro_counter_sensor.v           ;# v1 sensor (survives project-flow synth; v2 counter gets pruned)
add_files ../cw305/cw305_reg_bnn.v
add_files ../cw305/cw305_bnn_top.v
set HLS_RTL [glob -nocomplain ${HLS_RTL_DIR}/*.v]
if {[llength $HLS_RTL] == 0} { error "No HLS Verilog in '$HLS_RTL_DIR'" }
add_files $HLS_RTL

set_property top cw305_bnn_top [current_fileset]
set_property generic "KSIZE=$K" [current_fileset]

add_files -fileset constrs_1 $CWXDC
add_files -fileset constrs_1 ../cw305/cw305_bnn.xdc
add_files -fileset constrs_1 ../cw305/ro_sensor.xdc
set_property STEPS.PLACE_DESIGN.TCL.PRE   [file normalize ../cw305/ro_pblock.tcl]     [get_runs impl_1]
set_property STEPS.WRITE_BITSTREAM.TCL.PRE [file normalize ../cw305/allow_ro_loops.tcl] [get_runs impl_1]

update_compile_order -fileset sources_1
launch_runs synth_1 -jobs 4
wait_on_run synth_1
launch_runs impl_1 -to_step write_bitstream -jobs 4
wait_on_run impl_1
file copy -force ./vivado_${BUILD_NAME}/cw305_bnn_${BUILD_NAME}.runs/impl_1/cw305_bnn_top.bit \
                 ./cw305_bnn_${BUILD_NAME}.bit
puts "BITSTREAM: build/cw305_bnn_${BUILD_NAME}.bit"
exit
