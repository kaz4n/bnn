# P1 -- Vivado HLS 2016.4: csim (bit-exact verify) + csynth + export IP.
# Run:  vivado_hls -f run_hls.tcl -tclargs 3      (or 5 for Model 2)
# From a shell with C:\Xilinx\Vivado_HLS\2016.4\bin on PATH.

# K from env var KSIZE (robust across vivado_hls arg parsing). Default 3.
#   Windows:  set KSIZE=3   (or 5)   then   vivado_hls -f run_hls.tcl
set K 3
if {[info exists ::env(KSIZE)]} { set K $::env(KSIZE) }
set FANOUT 16
if {[info exists ::env(FANOUT)]} { set FANOUT $::env(FANOUT) }
set part xc7a100tftg256-2           ;# CW305-A100 = XC7A100T-2FTG256 (A35: xc7a35tftg256-2)

open_project bnn_conv1_hls_${K}_f${FANOUT}
set_top bnn_conv1
add_files        ../hls/bnn_conv1.cpp    -cflags "-DKSIZE=$K -DFANOUT=$FANOUT -I../hls"
add_files -tb    ../hls/bnn_conv1_tb.cpp  -cflags "-DKSIZE=$K -DFANOUT=$FANOUT -I../hls"

open_solution sol1
set_part $part
create_clock -period 40 -name default    ;# 25 MHz target clock (synchronous capture)

# vectors for csim -- generate first (from build/):
#   python ../sw_golden/golden_conv.py --ksize 3 --random --seed 3 --out ../hls/data_3
# absolute path so it resolves from the solution's csim/build working dir.
set datadir [file normalize ../hls/data_${K}]
csim_design -argv "$datadir"
csynth_design
export_design -format ip_catalog -rtl verilog
close_project
exit
