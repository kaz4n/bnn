# Hardware setup — running the experiment on CW305 + ChipWhisperer-Lite

Everything needed to go from the verified software (P1 conv, P2 weights, P3 attacks) to
**real measured power traces** and the paper's two attacks on silicon. This is P0 + the
hardware half of P1 + P3-on-real-traces.

---

## 1. Hardware you need

| Item | Notes |
|---|---|
| CW305 Artix-7 target board (XC7A100T) | have ✓ |
| ChipWhisperer-Lite / CW1173 | have ✓ (capture) |
| 2× USB cables | one to CW-Lite, one to CW305 (control + FPGA programming over USB) |
| 1× SMA cable — **power** | CW305 amplified shunt output (`X4`/`VOUT`) → CW-Lite **MEASURE** (LNA in) |
| 1× SMA/MCX cable — **clock** | CW305 clock out (`HS2`) → CW-Lite **HS-In** (enables SYNCHRONOUS capture) |
| Jumper wire — **trigger** | CW305 `tio_trigger` → CW-Lite **GPIO4/TRIG** (20-pin header) |

The 20-pin ChipWhisperer connector between the boards carries trigger + IO; the two SMAs
carry power (to MEASURE) and clock (to HS-In). The clock link is what makes capture
synchronous — the whole reason paper §5 collapses to a per-cycle reduction.

## 2. CW305 board configuration (jumpers/switches)

- **Power-measurement shunt**: set jumpers so VCC-INT is measured through the on-board
  shunt + amplifier, and that amplified node goes to the `X4` SMA. (CW305 manual:
  "measuring power".) Using the AMPLIFIED output is essential — raw shunt is ~mV, near
  noise (paper §5.1 makes the same point).
- **FPGA configuration mode**: USB (so ChipWhisperer programs the bitstream over USB).
- **Clock**: drive the FPGA from the on-board PLL (CDCE906), set low (see §5). Route the
  FPGA working clock to `HS2` out (our `tio_clkout`).
- **Bank/IO voltage** switches per CW305 defaults for the 20-pin IO.

## 3. Software

- **ChipWhisperer** (`pip install chipwhisperer`) + correct CW-Lite firmware.
- **Vivado 2016.4 + Vivado HLS 2016.4** (have ✓).
- **Stock CW305 reference sources** from the ChipWhisperer repo
  (`hardware/victims/cw305_artixtarget/fpga/common`): `cw305_usb_reg_fe.v`, `cw305_pll`,
  clock modules, `cw305_main.xdc`. Set `export CW305_SRC=<that folder>`.

## 4. Build the bitstream (P1)

```bash
# 4a. HLS: bit-exact csim + synth + export IP (do for K=3, then K=5)
export PATH="/c/Xilinx/Vivado_HLS/2016.4/bin:$PATH"
cd build && vivado_hls -f run_hls.tcl -tclargs 3      # -> PASS, exports IP

# 4b. After csynth, open the exported IP's RTL and copy the exact BRAM/ap_ctrl port
#     names into cw305/cw305_reg_bnn.v (HLS may name them img_address0/ce0/q0 etc.).

# 4c. Vivado: build bitstream for XC7A100T-2FTG256
export CW305_SRC=/path/to/cw305_artixtarget/fpga/common
vivado -mode batch -source run_vivado.tcl -tclargs 3  # -> cw305_bnn_top.bit
```
Repeat 4a/4c with `5` for Model 2. Sanity-check over USB (write image+kernel, GO, read
back output) against the HLS C-sim before trusting traces.

## 5. Capture power traces (P0 proven first, then real capture)

**P0 first (de-risk):** program a STOCK ChipWhisperer CW305 AES example, capture one
averaged trace. If that fails, fix wiring/clock/trigger before anything else.

Then capture for the BNN:
```bash
cd host
python cw305_bnn_capture.py --bitstream <cw305_bnn_top.bit> \
    --kernels ../training/artifacts/model_3x3/layer1_kernels.npy --ksize 3 \
    --n-kernels 9 --n-images 500 --avg 50 --freq 25e6 --out traces
```
Key capture settings (in the script): `scope.clock.adc_src="extclk_x4"` (synchronous,
4 samples/cycle), trigger `tio4`, target clock ~25 MHz. Writes `img####.npz` (one per
image, 9 kernels each).

**SNR — the real risk.** Layer-1 conv unit power is ~0.57 mW (paper Table 3). Mitigate:
- `--avg 50`+ (average many captures per image,kernel),
- low clock (~10–25 MHz),
- keep the rest of the fabric idle during the trigger window (only layer-1 runs),
- use the amplified shunt path.

## 6. Run the two paper algorithms on the real traces (P3)

No attack-code changes — just point the hardware driver at the captures:
```bash
cd attack
python run_on_hardware.py --ksize 3 --traces-dir ../host/traces \
    --n-profile 300 --n-eval 200 --out results_hw_3x3
python run_on_hardware.py --ksize 5 --traces-dir ../host/traces \
    --n-profile 300 --n-eval 200 --out results_hw_5x5
```
This runs **§6 background detection** and **§7 power template** (build from 300, attack
200, δ=1.0, 3 groups of 3), scores recognition with the golden MLP, and writes
`summary.json` + recovered images. Compare to paper: recog ≈ 81.6% (background) / 89.8%
(template) for 3×3.

Set `samples_per_cycle` = your `adc_mul` (4 for `extclk_x4`); the host already stores it.
Cycle 0 aligns to the trigger edge (trigger spans exactly the conv), so no §5-style
realignment is needed.

---

## Checklist (order, by risk)
1. ☐ Wire boards (power SMA, clock SMA, trigger jumper, 2 USB); set CW305 jumpers.
2. ☐ P0: capture stock CW305 AES trace — proves the bench.
3. ☐ Build P1 bitstream (HLS csim PASS → Vivado → .bit), fix HLS IP port names.
4. ☐ USB functional check: HW conv output == C-sim.
5. ☐ Capture 500 images × 9 kernels (`cw305_bnn_capture.py`), averaged.
6. ☐ `run_on_hardware.py` for K=3 and K=5 → real-silicon results vs paper.

Until step 5, `attack/evaluate.py` (simulated traces) is the stand-in — same algorithms.
