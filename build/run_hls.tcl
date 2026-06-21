# P1 -- Vivado HLS 2016.4: csim (bit-exact verify) + csynth + export IP.
# Run:  vivado_hls -f run_hls.tcl -tclargs 3      (or 5 for Model 2)
# From a shell with C:\Xilinx\Vivado_HLS\2016.4\bin on PATH.

set K 3
if {$argc >= 1} { set K [lindex $argv 0] }
set part xc7a100t-ftg256-2          ;# CW305 Artix-7 (XC7A100T-2FTG256)

open_project bnn_conv1_hls_${K}
set_top bnn_conv1
add_files        ../hls/bnn_conv1.cpp    -cflags "-DKSIZE=$K -I../hls"
add_files -tb    ../hls/bnn_conv1_tb.cpp  -cflags "-DKSIZE=$K -I../hls"

open_solution sol1
set_part $part
create_clock -period 40 -name default    ;# 25 MHz target clock (synchronous capture)

# vectors for csim (generate first: python ../sw_golden/golden_conv.py --ksize K --random)
# csim expects ./data/ relative to the solution's csim dir; copy or point tb arg.
csim_design -argv "../../hls/data_${K}"
csynth_design
export_design -format ip_catalog -rtl verilog
close_project
exit
