# Constraints for the RGB leakage design.
#
# Same shape as cw305_leakage_grey.xdc, and for the same reason: the binary design's XDC
# names the crypto clock net "U_reg/crypto_clk", which does not exist at that level, so
# synthesis logs
#   CRITICAL WARNING [Synth 8-3321] create_clock attempting to set clock on an
#   unknown port/pin
# and the constraint is silently dropped. The convolution logic is then timed against
# tio_clkin at 100 MHz from the stock CW305 XDC, which is not the speed the DUT ever runs
# at (5 MHz on the bench). Naming the real net makes the timing report mean something.
#
# 40 ns = 25 MHz, comfortably above the 5-10 MHz the bench uses, so a pass here leaves
# headroom for the clock-transfer conditions.
create_clock -period 40.000 -name leakage_crypto_clk [get_nets crypto_clk]

set_clock_groups -asynchronous \
  -group [get_clocks -include_generated_clocks usb_clk] \
  -group [get_clocks -include_generated_clocks leakage_crypto_clk]

# Image, kernel and the dataflow mode are written only while the engine is idle and are
# then static for the whole scan, so the USB-to-crypto handoff is not a timed path. The
# mode register additionally crosses through a two-flop synchronizer in the register file.
set_false_path -from [get_clocks usb_clk] -to [get_clocks leakage_crypto_clk]
set_false_path -from [get_clocks tio_clkin] -to [get_clocks usb_clk]
set_false_path -from [get_clocks usb_clk] -to [get_clocks tio_clkin]
