# Implement from the completed synth checkpoint (sensor present). In-process: no worker
# spawn, no re-synth. Applies co-location pblock + RO-loop allow + CRC-disable.
open_checkpoint vivado_ro3pb/cw305_bnn_ro3pb.runs/synth_1/cw305_bnn_top.dcp
# synth DCP carries timing but NOT physical (LOC/IOSTANDARD) constraints -> re-read the pin
# XDCs before place, else write_bitstream DRC fails (NSTD-1/UCIO-1 on usb_*/clk/trigger).
set CWXDC "C:/Users/narut/ChipWhisperer/chipwhisperer/firmware/fpgas/aes/vivado/cw305.xdc"
read_xdc $CWXDC
read_xdc ../cw305/cw305_bnn.xdc
read_xdc ../cw305/ro_sensor.xdc
set nloop [llength [get_nets -hier -quiet *ro_loop*]]
puts "IMPL: $nloop ro_loop nets in synth DCP"
set_property ALLOW_COMBINATORIAL_LOOPS TRUE [get_nets -hier -quiet *ro_loop*]
if {[catch {source ../cw305/ro_pblock.tcl} e]} { puts "WARNING pblock skipped: $e" }
opt_design
place_design
route_design
source ../cw305/allow_ro_loops.tcl
write_bitstream -force cw305_bnn_ro3pb.bit
puts "BITSTREAM: build/cw305_bnn_ro3pb.bit"
exit
