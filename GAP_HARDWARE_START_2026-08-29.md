# Hardware-first start on the input-recovery research gap

Date: 2026-08-29  
Bench: CW-Lite + CW305-A100, controlled Khalifa University lab setup

## Main result today

I did not accept any BNN or Power2Picture capture as valid today. The reason is important: the CW305 is still answering like the stock AES design after attempts to program the custom leakage bitstream. Because of this, the new scripts now refuse to capture traces if the custom design is not really active on the FPGA.

This is a good failure. It stops us from training or reporting on traces that only came from the stock AES bitstream or from a host-side path.

## What I verified on real hardware

### 1. The ChipWhisperer analog capture chain works

I ran the stock AES hardware self-test:

```powershell
cd C:\Users\narut\OneDrive\Desktop\Project\bnn\host
C:\Users\narut\ChipWhisperer\cwenv\Scripts\python.exe cw305_p0_selftest.py
```

Result:

- CW-Lite connected.
- CW305 connected.
- ADC locked: `True`.
- Real trace captured: `5000` samples.
- Trace range: `-0.0654 .. 0.0244`.
- AES ciphertext: `5aea53c65894c3f145f06e41b50b940a`.

Saved files:

- `host/p0_trace.npy` is the raw trace saved by the stock AES self-test.
- `report/hardware_aes_control_trace_pass_20260829.npz`
- `report/hardware_aes_control_trace_pass_20260829.png`
- `report/hardware_aes_control_trace_pass_20260829.json`

This proves the CW-Lite/CW305 capture path can collect a real power trace.

![Valid stock AES control trace](report/hardware_aes_control_trace_pass_20260829.png)

### 2. The custom leakage bitstream builds

The custom leakage design built successfully with Vivado 2016.4:

- Bitstream: `build/cw305_leakage_d8_l63.bit`
- SHA-256 of last successful bitstream: `A5BB029E693174A3A6E63384BA5BAA1F464DE675C9DC1C16DAF290087BABF353`
- Design in header: `cw305_leakage_top`
- Build time in header: `2026/08/29 23:38:50`
- Routed setup slack: positive, about `WNS=2.595`
- Routed TNS: `0.000`

This means the HDL is not failing because of FPGA size or timing.

### 3. The CW305 is not activating the custom bitstream right now

I tested programming/reading several ways:

- `cw.target(..., bsfile=custom_bitstream, force=True)`
- manual `target.fpga.FPGAProgram(...)`
- slower programming speed
- programming from the actual bitstream sync-word offset
- erase + program custom again

After these attempts, the board still read:

- page 2: `0x02`
- page 3: `0x05`
- page 4: `0x2e`

Those values match the stock CW305 AES register map: crypto type, crypto revision, and identify value. Also, page behavior matched AES registers: page 5 acted like AES GO/status and page 6 acted like AES text input.

So the honest conclusion is:

> The custom leakage bitstream file exists and builds, but the active FPGA image on the board is still stock AES.

My confidence is high for this conclusion because it comes from live hardware readback, not simulation.

## Changes made to avoid false hardware results

### Capture scripts now reject stock AES

Updated:

- `host/cw305_leakage_func_check.py`
- `host/cw305_leakage_capture.py`
- `host/cw305_p2p_capture.py`

They now read pages 2, 3, and 4 before capture. If they see the AES signature `0x02, 0x05, 0x2e`, they stop with a clear error:

```text
CW305 is still responding as the stock AES design...
Check S1 mode switches: USB programming requires M0=1, M1=1, M2=1
```

This is important for the paper work. We should not allow a capture folder to exist unless the FPGA is actually running our leakage design.

### Capture scripts now use safer small FPGA transfers

I added small control-transfer helpers for image, kernel, status, GO, and output reads. This avoids silently switching to ChipWhisperer bulk transfers for long image/output regions. The current CW305 firmware reports as old:

```text
firmware 0.51.0, latest 0.54.0
```

So small verified transfers are safer for the first real hardware baseline.

### Power2Picture training now has a hardware gate

Updated:

- `attack/p2p_generator.py`

New option:

```powershell
--require-hardware
```

This refuses to train if the capture manifest looks simulated, has no hardware provenance, or has a failed functional check.

I also changed the default split to random instead of sequential. This avoids a weak result where the model learns capture order, temperature drift, or time drift instead of input leakage.

### Added capture validator

Added:

- `attack/validate_hardware_capture.py`

This checks a capture directory before training:

- manifest says live CW305/ChipWhisperer capture,
- trace source is not simulation,
- bitstream hash exists,
- ADC lock exists,
- functional check passed,
- trace arrays are finite and not constant.

## Why the older implementation stalled or gave low detection

The old flow allowed a dangerous situation:

1. The script could call the CW305 programming API.
2. ChipWhisperer could return DONE high.
3. The script could continue.
4. But the board could still be running stock AES or another wrong FPGA image.

If that happens, the BNN host code is writing MNIST images and kernels to addresses that only make sense for the custom leakage design. On stock AES, those same pages mean AES clock settings, user LED, type/revision registers, GO/status, and text input. Then the captured trace is not a BNN trace at all.

That explains why reconstruction/detection was weak or inconsistent: the learning code could be looking at power from the wrong workload.

## Current blocker

The blocker is physical/programming state, not the reconstruction algorithm.

Most likely causes:

1. CW305 S1 mode switches are not in USB programming mode.
2. The board is reloading the stock AES image from SPI flash after programming/erase.
3. The old NAEUSB firmware is causing misleading programming status.
4. Less likely: board/jumper/power issue around FPGA configuration.

The first thing to check physically:

```text
CW305 S1 mode switches:
M0 = 1
M1 = 1
M2 = 1
```

NewAE documents this S1 setting as the CW305 USB configuration mode:
https://rtfm.newae.com/Targets/CW305%20Artix%20FPGA/

Then power-cycle the CW305 or press USB RST/SW3 and rerun:

```powershell
cd C:\Users\narut\OneDrive\Desktop\Project\bnn\host
C:\Users\narut\ChipWhisperer\cwenv\Scripts\python.exe cw305_leakage_func_check.py `
  --bitstream ..\build\cw305_leakage_d8_l63.bit `
  --images mnist_test.npz `
  --kernels ..\training\artifacts\model_3x3\layer1_kernels.npy `
  --probe-kernels onehot `
  --img 0 `
  --kidx 0 `
  --clock-source cw_lite
```

Expected next valid result:

- pages 2/3/4 must not read `0x02, 0x05, 0x2e`;
- functional check must report `PASS: 676 outputs bit-exact`;
- only then should we capture BNN traces.

## Next methodology after the board mode is fixed

I would run the work in this order:

1. **Hardware identity and custom-design proof**
   - prove CW-Lite serial and CW305 serial,
   - prove bitstream SHA-256,
   - prove not stock AES,
   - prove `676` output values match CPU golden output.

2. **Active/power-template baseline**
   - use one-hot `3x3` probe kernels,
   - randomize image order,
   - average a small number of repeated traces,
   - reconstruct image pixels from the power templates,
   - report pixel accuracy, F1, MSSIM, and digit recognizer accuracy.

3. **Power2Picture-style baseline**
   - capture one real trained-kernel trace per image,
   - require hardware manifest,
   - train with random train/val/test split,
   - report only metrics from capture folders passing the hardware validator.

4. **Novel gap direction**
   - after these baselines work, vary clock source, temperature/load, dwell cycles, and leakage lanes;
   - show how much reconstruction transfers or breaks under hardware condition changes;
   - this is closer to a publishable gap than only repeating the old paper.

## Current confidence

- Capture chain works on real hardware: **high confidence**.
- Custom bitstream builds and meets timing: **high confidence**.
- Custom bitstream is not currently active on the FPGA: **high confidence**.
- Cause is S1 mode / boot-from-flash / programming mode: **medium confidence**, because I cannot physically see the switches from software.
- BNN/Power2Picture reconstruction from current hardware today: **not claimed**.
