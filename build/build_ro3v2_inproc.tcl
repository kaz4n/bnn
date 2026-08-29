# In-process (non-project) build of the ro3v2 bitstream. Runs synth/place/route/bitstream
# in ONE vivado process -- no launch_runs worker spawn (that fails when launched headless
# from a background shell, stalling at the synth queue). v2 sensor + co-location pblock.
#   KSIZE=3 FANOUT=16 vivado -mode batch -source build_ro3v2_inproc.tcl

set K 3;      if {[info exists ::env(KSIZE)]}  { set K $::env(KSIZE) }
set FANOUT 16; if {[info exists ::env(FANOUT)]} { set FANOUT $::env(FANOUT) }
set part xc7a100tftg256-2
set HLS_RTL_DIR bnn_conv1_hls_${K}_f${FANOUT}/sol1/impl/verilog
if {![file isdirectory $HLS_RTL_DIR]} { error "Missing HLS RTL dir '$HLS_RTL_DIR'" }
set CWHDL "C:/Users/narut/ChipWhisperer/chipwhisperer/firmware/fpgas/aes/hdl"
set CWXDC "C:/Users/narut/ChipWhisperer/chipwhisperer/firmware/fpgas/aes/vivado/cw305.xdc"
if {[info exists ::env(CW305_SRC)]} { set CWHDL $::env(CW305_SRC) }

read_verilog "$CWHDL/cw305_usb_reg_fe.v"
read_verilog "$CWHDL/clocks.v"
read_verilog ../cw305/cdc_pulse.v
read_verilog ../cw305/ro_counter_sensor.v      ;# v1 sensor (survives synth; v2 counter got pruned)
read_verilog ../cw305/cw305_reg_bnn.v
read_verilog ../cw305/cw305_bnn_top.v
foreach f [glob -nocomplain ${HLS_RTL_DIR}/*.v] { read_verilog $f }
read_xdc $CWXDC
read_xdc ../cw305/cw305_bnn.xdc

synth_design -top cw305_bnn_top -part $part -generic KSIZE=$K
# dump hierarchical cell names so the pblock can be anchored on real names (diagnostic)
set fh [open cells_dump.txt w]
foreach c [get_cells -hier -filter {IS_PRIMITIVE==0}] { puts $fh $c }
close $fh
puts "CELLS_DUMP: [llength [get_cells -hier -filter {IS_PRIMITIVE==0}]] hierarchical cells -> cells_dump.txt"
# co-locate sensor + conv (dilution/locality ceiling-raiser). OPTIONAL: a pblock failure
# must NOT abort the build -- the v2 sensor alone already tests the counter/S6 fix.
if {[catch {source ../cw305/ro_pblock.tcl} emsg]} {
    puts "WARNING: pblock skipped ($emsg) -- building v2 sensor WITHOUT co-location"
}

opt_design
place_design
route_design
report_timing_summary -quiet -file timing_ro3v2.rpt
# RO comb-loop allow + CRC-disable MUST run here (routed nets exist; CRC-disable lets the
# CW305 load a bitstream whose RO loops break readback CRC). Same proven hook as the
# working ro3 build.
source ../cw305/allow_ro_loops.tcl
write_bitstream -force cw305_bnn_ro3pb.bit
puts "BITSTREAM: build/cw305_bnn_ro3pb.bit"
exit
