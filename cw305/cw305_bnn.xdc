# P1 -- CW305 constraint overrides for the BNN conv design.
# Reuse the STOCK cw305_main.xdc for the full Artix-7 pin map (USB bus, PLL, LEDs,
# tio_trigger, tio_clkout). This file only adds clocking/timing for our design.
#
# The stock cw305_main.xdc already constrains the top-level port names used in
# cw305_bnn_top.v (usb_clk, usb_data, usb_addr, usb_rdn/wrn/cen, usb_trigger,
# pll_clk1, tio_trigger, tio_clkout, leds). Do NOT duplicate those here.

# Target/working clock: keep low so ChipWhisperer-Lite synchronous capture is clean.
# 25 MHz example (SETUP_PLAN.md C3 recommends ~10-25 MHz). Adjust pll_clk1 source freq
# in cw305_pll accordingly; CW-Lite samples tio_clkout (HS-In) at adc_mul x this.
create_clock -period 40.000 -name crypt_clk [get_nets U_*/crypt_clk]

# usb_clk is constrained by stock xdc (~30 MHz). Mark the two domains asynchronous;
# all crossings use ASYNC_REG synchronizers in cw305_reg_bnn.v.
set_clock_groups -asynchronous \
  -group [get_clocks -include_generated_clocks usb_clk] \
  -group [get_clocks -include_generated_clocks crypt_clk]
