# P1 — Layer-1 conv unit (HLS) + CW305 integration

Reproduces the attacked first conv layer (bnn-fpga line buffer) on CW305 + CW-Lite.
See `../SETUP_PLAN.md` Phase 1 for context.

## What is verified NOW (no hardware needed)

`bnn_conv1.h` / `.cpp` — the conv unit. `bnn_conv1_tb.cpp` + `../sw_golden/golden_conv.py`
prove it **bit-exact** vs a numpy 'same' convolution, including the P2 kernel bit-packing
convention, for both 3×3 and 5×5. Run:

```bash
# uses g++ bundled with Vivado HLS (C:\Xilinx\Vivado_HLS\2016.4\msys\bin)
export PATH="/c/Xilinx/Vivado_HLS/2016.4/msys/bin:$PATH"
for K in 3 5; do
  python ../sw_golden/golden_conv.py --ksize $K --random --seed $K --out data_$K
  g++ -DKSIZE=$K -O2 -I . -o tb_$K.exe bnn_conv1.cpp bnn_conv1_tb.cpp
  ./tb_$K.exe data_$K
done
# -> PASS: KSIZE=3, all 784 outputs bit-exact vs golden   (and KSIZE=5)
```

To check a real trained kernel after P2 finishes:
`python ../sw_golden/golden_conv.py --ksize 3 --kernels ../training/artifacts/model_3x3/layer1_kernels.npy --kidx 0 --out data_real && ./tb_3.exe data_real`

## What is a HARDWARE-GATED template (untested, needs bench + Vivado)

- `../cw305/cw305_bnn_top.v`, `cw305_reg_bnn.v` — CW305 top + register module wrapping
  the HLS IP. Drives `tio_trigger` for exactly the conv (ap_start..ap_done) and
  `tio_clkout` to CW-Lite HS-In for synchronous capture.
- `../cw305/cw305_bnn.xdc` — clock/CDC overrides (reuse stock `cw305_main.xdc` for pins).
- `../build/run_hls.tcl` — csim + csynth + export IP (`vivado_hls -f run_hls.tcl -tclargs 3`).
- `../build/run_vivado.tcl` — build bitstream for XC7A100T-2FTG256.
- `../host/cw305_bnn_capture.py` — ChipWhisperer capture → `.npz` traces.

### You must supply (not in repo)
- Stock ChipWhisperer CW305 sources: `cw305_usb_reg_fe.v`, `cw305_pll.v`, clock modules,
  `cw305_main.xdc` (from `chipwhisperer/hardware/victims/cw305_artixtarget/fpga/common`).
  Set `CW305_SRC` env var to that folder before `run_vivado.tcl`.
- After `csynth`, confirm the exported IP's BRAM/`ap_ctrl_hs` port names match the instance
  in `cw305_reg_bnn.v` (HLS sometimes names ports `img_address0/ce0/q0`, etc.) and fix.

## Key faithfulness / mapping notes
- 1 pixel/cycle, line size 28, 'same' padding (output 28×28) — matches paper Table 1 /
  line buffer; the per-cycle ↔ 12-pixel-window mapping the attack needs is intact.
- One kernel per invocation: the attack captures one trace per kernel (paper: 64 total,
  9 used for the template).
- ⚠️ Kernel bit-order in `*_packed.bin` is a convention shared by P2 export, the HLS
  unpacker, and the host. Don't change one without the others. The csim test guards it.
- Without a bench, drive the whole attack (P3) from `../attack/trace_sim.py`, which
  emits the same `.npz` the host would.
