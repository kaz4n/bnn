# Resume this experiment's routed checkpoint for physical hold repair.
set root [file dirname [file dirname [file normalize [info script]]]]
set out [file join $root build]
open_checkpoint [file join $out implemented.dcp]
phys_opt_design -hold_fix
route_design
report_timing_summary -file [file join $out timing.rpt]
report_utilization -file [file join $out utilization.rpt]
report_drc -file [file join $out drc.rpt]
write_checkpoint -force [file join $out implemented.dcp]
foreach delay {max min} {
    set path [get_timing_paths -delay_type $delay -max_paths 1 -nworst 1]
    if {[llength $path] == 0 || [get_property SLACK $path] < 0} { error "Timing $delay failed; no bitstream" }
}
write_bitstream -force [file join $out cw305_rgb_instrumented.bit]
