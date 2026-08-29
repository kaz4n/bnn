# Applied just before write_bitstream, when routed RO nets exist.
set ro_loop_nets [get_nets -hierarchical -quiet *ro_loop*]
puts "RO_SENSOR: allowing combinational loops on [llength $ro_loop_nets] ro_loop nets"
if {[llength $ro_loop_nets] > 0} {
    set_property ALLOW_COMBINATORIAL_LOOPS TRUE $ro_loop_nets
}
# Ring-oscillator combinational loops make the post-config CRC/readback inconsistent, so
# the CW305 SAM3U treats the load as failed. Disable the bitstream CRC so it loads.
set_property BITSTREAM.GENERAL.CRC DISABLE [current_design]
