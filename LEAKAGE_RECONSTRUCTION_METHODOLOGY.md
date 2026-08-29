# Leakage-First MNIST Reconstruction Methodology

This is the hardware-adapted path for reproducing the idea of `1803.05847v2.pdf`
on the current CW305 + ChipWhisperer-Lite bench. The original paper used a much
higher-bandwidth external power setup. Here the target is slowed to 5 MHz and
the CW-Lite ADC samples synchronously at `extclk_x4`, giving 4 samples per FPGA
clock and a stable 32-sample window for each 3x3 patch.

## Hardware Design

Files:

- `cw305/mnist_leakage_core.sv`
- `cw305/cw305_reg_leakage_demo.sv`
- `cw305/cw305_leakage_top.sv`
- `cw305/cw305_leakage.xdc`
- `build/run_vivado_leakage.tcl`

Design:

- 28x28 binary MNIST image stored over CW305 USB registers.
- 3x3 binary convolution over 26x26 valid windows.
- Each window held for `DWELL=8` FPGA clocks.
- Retained `DONT_TOUCH/KEEP` leak bank toggles as `0 -> XNOR(window,kernel) -> 0`.
- Conv score is still written back as signed 8-bit `2*matches - 9`.
- Default build uses `LEAK_LANES=63`, producing 567 retained data-dependent FFs.

Build/verification:

```powershell
cd build
C:\Xilinx\Vivado\2016.4\bin\vivado.bat -mode batch -source run_vivado_leakage.tcl
```

Observed `d8_l63` build:

- Bitstream: `build/cw305_leakage_d8_l63.bit`
- Timing: WNS `3.326 ns`, TNS `0.000`, all user constraints met.
- Utilization: 2942 LUTs, 1470 FFs, 0 BRAM.
- Clock interaction: USB and crypto clocks async-grouped.
- XSim core test: 676 writes, 5408 busy cycles.

## Reconstruction Method

Use 9 one-hot probe kernels, one per position in the 3x3 patch. This keeps the
same binary-conv datapath but makes each patch identifiable from leakage.

For one-hot kernel `i`, the match count is:

```text
count_i = 8 - total_ones + 2 * bit_i
```

Across all 9 probe kernels:

```text
total_ones = (72 - sum(count_i)) / 7
bit_i = (count_i - 8 + total_ones) / 2
```

The implemented attack uses profiled templates instead of directly assuming
linear count measurements:

- Capture known profile images.
- Extract per-window `mean_abs` features from each 32-sample window.
- Learn per-kernel lookup tables for match counts 0..9.
- Decode each eval window by nearest template among all 512 possible 3x3 patches.
- Majority-vote overlapping patches to reconstruct the full 28x28 binary image.

## Commands Used

Functional silicon check:

```powershell
cd host
C:\Users\narut\ChipWhisperer\cwenv\Scripts\python.exe cw305_leakage_func_check.py `
  --bitstream ..\build\cw305_leakage_d8_l63.bit `
  --images mnist_test.npz `
  --probe-kernels onehot `
  --kidx 0 `
  --img 0 `
  --freq 5000000
```

Result:

```text
PASS: 676 outputs bit-exact
```

Hardware capture:

```powershell
cd host
C:\Users\narut\ChipWhisperer\cwenv\Scripts\python.exe cw305_leakage_capture.py `
  --bitstream ..\build\cw305_leakage_d8_l63.bit `
  --images mnist_test.npz `
  --out traces_leakage_60_l63 `
  --n-images 60 `
  --n-kernels 9 `
  --avg 5 `
  --probe-kernels onehot `
  --freq 5000000 `
  --dwell 8 `
  --gain 40 `
  --force
```

Attack:

```powershell
cd attack
C:\Users\narut\ChipWhisperer\cwenv\Scripts\python.exe leakage_first_reconstruct.py `
  --traces ..\host\traces_leakage_60_l63 `
  --out results_leakage_60_l63_p30 `
  --profile-count 30 `
  --eval-count 30 `
  --probe-kernels onehot `
  --feature mean_abs
```

Hardware result over 30 held-out eval images:

```json
{
  "bit_acc": 0.984141156462585,
  "foreground_precision": 0.9139006231015527,
  "foreground_recall": 0.9412094863118654,
  "foreground_f1": 0.9236224212163883,
  "foreground_iou": 0.860143164232035,
  "all_zero_bit_acc": 0.891326530612245,
  "patch_exact": 0.4955128205128205
}
```

Visual check:

- `report/leakage_reconstruction_montage.png`

## Interpretation

Yes, the input can be reconstructed from side-channel information on this
hardware. The reconstruction is not paper-faithful in power-equipment terms: it
is hardware-adapted. The key change is to replace the original high-bandwidth
measurement assumptions with a slow, synchronous, leakage-amplified CW305
experiment.

This is suitable for a research demo if described honestly:

- Baseline: original trained-kernel attack is weak on CW-Lite due bandwidth,
  capture depth, and board/noise differences.
- Adaptation: controlled probe kernels and retained datapath switching make the
  side channel measurable on CW305/CW-Lite.
- Evidence: bit-exact hardware functionality, timing-clean bitstream, real
  power captures, and held-out image reconstruction above background baseline.

