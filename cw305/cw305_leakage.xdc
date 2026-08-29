# Leakage demo crypto clock. Stock CW305 XDC supplies pins and board clocks.
create_clock -period 40.000 -name leakage_crypto_clk [get_nets U_reg/crypto_clk]

set_clock_groups -asynchronous \
  -group [get_clocks -include_generated_clocks usb_clk] \
  -group [get_clocks -include_generated_clocks leakage_crypto_clk]

# Image/kernel registers are written only while engine is idle, then remain static.
set_false_path -from [get_clocks usb_clk] -to [get_clocks leakage_crypto_clk]
