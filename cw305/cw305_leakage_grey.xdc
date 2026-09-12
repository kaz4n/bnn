# Constraints for the grey-pixel leakage design.
#
# The binary design's XDC names the crypto clock net "U_reg/crypto_clk", which
# does not exist at that level, so synthesis logs
#   CRITICAL WARNING [Synth 8-3321] create_clock attempting to set clock on an
#   unknown port/pin
# and the constraint is silently dropped.  The convolution logic then gets timed
# against tio_clkin at 100 MHz from the stock CW305 XDC, which is not the speed
# the DUT ever runs at (5 MHz on the bench, 10 MHz in the clock-transfer tests).
# Naming the real net makes the report mean something.
create_clock -period 40.000 -name leakage_crypto_clk [get_nets crypto_clk]

set_clock_groups -asynchronous \
  -group [get_clocks -include_generated_clocks usb_clk] \
  -group [get_clocks -include_generated_clocks leakage_crypto_clk]

# Image and kernel are written only while the engine is idle and are then static
# for the whole run, so the USB-to-crypto handoff is not a timed path.
set_false_path -from [get_clocks usb_clk] -to [get_clocks leakage_crypto_clk]
set_false_path -from [get_clocks tio_clkin] -to [get_clocks usb_clk]
set_false_path -from [get_clocks usb_clk] -to [get_clocks tio_clkin]
