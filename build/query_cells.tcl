open_checkpoint vivado_ro3pb/cw305_bnn_ro3pb.runs/synth_1/cw305_bnn_top.dcp
puts "=== match counts ==="
foreach pat {*ro_sensor* *ro_loop* *tap* *U_RO* *sensor* *sum_delta* *U_inv* *RO*} {
    puts "  $pat : [llength [get_cells -hier -quiet -filter "NAME =~ $pat"]]"
}
puts "=== hierarchical cells under U_reg_bnn ==="
foreach c [get_cells -quiet -filter {IS_PRIMITIVE==0} U_reg_bnn/*] { puts "  $c" }
puts "=== any cell name containing 'ro' (first 20) ==="
set n 0
foreach c [get_cells -hier -quiet] {
    if {[string match -nocase *ro_* $c] || [string match -nocase *sensor* $c]} {
        puts "  $c"; incr n; if {$n>=20} break
    }
}
exit
