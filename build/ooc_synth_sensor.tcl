# Out-of-context synthesis of the v2 RO sensor -- de-risk before a full rebuild.
# Checks: it synthesizes, Vivado keeps the ROs + per-RO counters (DONT_TOUCH), and
# reports utilization. Fast (~minutes), no place/route/bitstream.
read_verilog ../cw305/ro_counter_sensor_v2.v
read_xdc     ../cw305/ro_sensor.xdc
synth_design -top ro_counter_sensor -part xc7a100tftg256-2 -mode out_of_context
report_utilization -hierarchical
# count retained ring-osc LUTs + counter FFs
puts "=== RO loop LUTs kept: [llength [get_cells -hier -quiet -filter {NAME =~ *U_inv*}]] ==="
puts "=== counter/bin FFs kept: [llength [get_cells -hier -quiet -filter {NAME =~ *bin_reg*}]] ==="
exit
