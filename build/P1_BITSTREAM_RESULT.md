# P1 — CW305 BNN bitstream BUILT ✅

`vivado -mode batch -source run_vivado.tcl` (KSIZE=3) → **`build/cw305_bnn_3.bit`** (2.2 MB).

- synth_design ✓, place_design ✓, route_design ✓, **write_bitstream completed successfully**
- 0 ERRORs. Integration synthesized/placed/routed clean.

## Design as built
- Stock CW305 infra (`cw305_usb_reg_fe.v`, `clocks.v`, `cw305.xdc`) + `cdc_pulse.v` +
  `cw305_bnn_top.v` (stock top, AES→BNN swap) + `cw305_reg_bnn.v` + HLS conv RTL (II=1).
- crypt_clk = `clocks.v` BUFGMUX **passthrough of PLL1** (no multiply) → frequency = host
  `target.pll.pll_outfreq_set(...)`. tio_clkout (ODDR) → CW-Lite HS-In for synchronous capture.
- Register map (host `fpga_write/read` addr): IMAGE=0 (784 B), OUTPUT=16 (1568 B int16 LE),
  KERNEL=32, GO=33, STATUS=34. trigger = conv output-write (II=1 → 784 cycles).

## ⚠️ Clock ceiling
Timing report WNS = −7.77 ns **against the stock XDC's fast clock** — the conv's combinational
K×K MAC path ≈ **17.7 ns → max ≈ 56 MHz**. Run the PLL at **≤25 MHz** (we use 10–25 MHz for
clean synchronous capture anyway), so this is met with margin. Do NOT clock the BNN above ~50 MHz.
(Optional: add an XDC `create_clock -period 40` override on the crypt path to make the timing
report green; not required for function.)

## Next (on bench)
1. `python host/dump_mnist.py` → `mnist_test.npz` (copy to host/).
2. cwenv: `python host/cw305_bnn_capture.py --bitstream build/cw305_bnn_3.bit \
   --kernels training/artifacts/model_3x3/layer1_kernels.npy --ksize 3 --freq 10e6`
   - first do a **functional check**: write one image+kernel, GO, `fpga_read(16,1568)`,
     compare to `sw_golden/golden_conv.py` (bit-exact) before trusting power traces.
3. `python attack/run_on_hardware.py --ksize 3 --traces-dir host/traces`.

K=5: rerun `run_vivado.tcl` with `set KSIZE=5`.
