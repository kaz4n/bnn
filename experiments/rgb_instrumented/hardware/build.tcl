set here [file dirname [file normalize [info script]]]
set root [file dirname $here]
set out [file join $root build]
file mkdir $out
set frontend [file normalize {C:/Users/narut/ChipWhisperer/chipwhisperer/firmware/fpgas/aes/hdl/cw305_usb_reg_fe.v}]
if {[info exists ::env(CW305_USB_FRONTEND)]} { set frontend $::env(CW305_USB_FRONTEND) }
if {![file exists $frontend]} { error "CW305 frontend source not found: $frontend" }
create_project -in_memory -part xc7a100tftg256-2
read_verilog $frontend
read_verilog -sv [file join $root rtl rgb_bayer_capture.sv]
read_verilog -sv [file join $here cw305_rgb_top.sv]
read_xdc [file join $here cw305_rgb.xdc]
synth_design -top cw305_rgb_top -part xc7a100tftg256-2
opt_design
place_design
route_design
phys_opt_design -hold_fix
route_design
report_timing_summary -file [file join $out timing.rpt]
report_utilization -file [file join $out utilization.rpt]
report_drc -file [file join $out drc.rpt]
# Preserve routed evidence even when a timing check rejects bitstream creation.
write_checkpoint -force [file join $out implemented.dcp]
set worst [get_timing_paths -max_paths 1 -nworst 1]
if {[llength $worst] == 0} { error "No timing path returned" }
if {[get_property SLACK $worst] < 0} { error "Timing failure; bitstream not written" }
set shortest [get_timing_paths -delay_type min -max_paths 1 -nworst 1]
if {[llength $shortest] == 0} { error "No hold timing path returned" }
if {[get_property SLACK $shortest] < 0} { error "Hold timing failure; bitstream not written" }
write_bitstream -force [file join $out cw305_rgb_instrumented.bit]
