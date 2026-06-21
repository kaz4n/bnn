# P1 -- Vivado 2016.4: build CW305 bitstream wrapping the HLS conv IP.
# Run:  vivado -mode batch -source run_vivado.tcl -tclargs 3   (or 5)
#
# PREREQUISITES (you must supply -- not in this repo):
#   $CW305_SRC : path to ChipWhisperer cw305_artixtarget/fpga/common stock sources
#                (cw305_usb_reg_fe.v, cw305_pll.v, clocks, cw305_main.xdc)
#   HLS IP exported by run_hls.tcl (bnn_conv1_hls_<K>/sol1/impl/ip)

set K 3
if {$argc >= 1} { set K [lindex $argv 0] }
set part xc7a100t-ftg256-2
set CW305_SRC $::env(CW305_SRC)        ;# export CW305_SRC=...\fpga\common

create_project cw305_bnn_${K} ./vivado_${K} -part $part -force

# HLS IP repo
set_property ip_repo_paths [list ../bnn_conv1_hls_${K}/sol1/impl/ip] [current_project]
update_ip_catalog

# sources: stock CW305 + our wrappers
add_files [glob $CW305_SRC/*.v]
add_files ../cw305/cw305_reg_bnn.v
add_files ../cw305/cw305_bnn_top.v
set_property top cw305_bnn_top [current_fileset]
set_property generic "KSIZE=$K" [current_fileset]

# constraints: stock pin map + our overrides
add_files -fileset constrs_1 $CW305_SRC/cw305_main.xdc
add_files -fileset constrs_1 ../cw305/cw305_bnn.xdc

# instantiate HLS IP (creates bnn_conv1 module used by cw305_reg_bnn)
# (the exported IP is added to the catalog; ensure its module name is `bnn_conv1`)

launch_runs synth_1 -jobs 4
wait_on_run synth_1
launch_runs impl_1 -to_step write_bitstream -jobs 4
wait_on_run impl_1
puts "bitstream: ./vivado_${K}/cw305_bnn_${K}.runs/impl_1/cw305_bnn_top.bit"
exit
