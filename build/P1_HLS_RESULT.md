# P1 — HLS synthesis result (layer-1 conv unit)

Vivado HLS 2016.4 run of the layer-1 binarized conv (`hls/bnn_conv1.cpp`) for both models.

- **Date:** 2026-06-23
- **Run:** `set KSIZE=3 & vivado_hls -f build/run_hls.tcl` (and `KSIZE=5`)
- **Part:** `xc7a100tftg256-2` (CW305-A100)

## Result: csim PASS (bit-exact), RTL generated ✅

```
PASS: KSIZE=3, all 784 outputs bit-exact vs golden
INFO: [SIM 211-1] CSim done with 0 errors.
INFO: [RTGEN 206-100] Finished creating RTL model for 'bnn_conv1'.
...
PASS: KSIZE=5, all 784 outputs bit-exact vs golden
```

RTL emitted: `bnn_conv1_hls_<K>/sol1/impl/verilog/{bnn_conv1,bnn_conv1_core,bnn_unpack_kernel}.v`

### Synthesis summary (K=3, 25 MHz target)
| Item | Value |
|---|---|
| Timing (estimated) | 8.14 ns (target 40 ns — easily met) |
| Resources | 692 LUT, 332 FF, 0 DSP, 0 BRAM |
| Latency | 20462 cycles / feature map |

### Port names match the wrapper ✅
HLS generated exactly the ports `cw305_reg_bnn.v` instantiates: `ap_clk/ap_rst/ap_start/
ap_done`, `img_Addr_A/EN_A/Dout_A`, `packed_kern_Addr_A/EN_A/Dout_A`, `out_r_Addr_A/EN_A/
WEN_A/Din_A`. (Minor: `*_WEN_A` is a 2-bit byte-enable on the 16-bit `out` port — tie/handle
in the wrapper.)

The IP-packaging step (`ap_source` error) is irrelevant — we add the raw `.v` files to
Vivado directly (`run_vivado.tcl` updated).

## Pipelined to II=1 ✅ (1 pixel/cycle = paper line buffer)

Rewrote `bnn_conv1.cpp`: preload image into fully-partitioned registers, then a
`PIPELINE II=1` raster loop emitting **one output per cycle**.

```
Pipelining result: Target II: 1, Final II: 1   (both preload and conv loops)
PASS: KSIZE=3, all 784 outputs bit-exact vs golden
PASS: KSIZE=5, all 784 outputs bit-exact vs golden
```

Latency (K=3): preload 785 + **conv 784 cycles** (1 output/cycle) = 1593 total.

Consequences (both now resolved):
1. **Attack cycle-mapping** — conv is now exactly 1 pixel/cycle, raster order, matching
   `attack/common.py` (`cyc_to_yx`) and the paper line buffer. Real traces will line up.
2. **Capture buffer** — conv = 784 cycles fits CW-Lite's 24573-sample buffer even at **x4**
   (784×4 = 3136 samples) → use `adc_src="extclk_x4"` for 4 samples/cycle resolution.
3. **Trigger** — `cw305_reg_bnn.v` asserts `tio_trigger` only while the conv loop writes
   `out` (II=1 → every cycle), so the captured window = exactly the 784 conv cycles, NOT
   the preload.

## Status
- ✅ Conv bit-exact (both K), **II=1 / 1 pixel-per-cycle line buffer**, RTL + ports good,
  trigger spans conv only, Vivado flow wired (raw `.v`).
- ▶️ Next: `run_vivado.tcl` with stock CW305 sources (`CW305_SRC`) → `cw305_bnn_top.bit`.
