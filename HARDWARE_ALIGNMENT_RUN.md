# Hardware Alignment Run - 2026-07-04

## What Was Hardware

- Fresh live capture was run from the CW305 + CW-Lite bench:
  - `host/traces_analog_live_500`
  - 500 MNIST test images
  - 9 kernels
  - 20 averages per image/kernel
  - CW305 bitstream: `build/cw305_bnn_3.bit`
  - Capture script: `host/cw305_analog_capture.py`
- The previous `results_analog_curvefit_full` run was offline processing of existing
  hardware-captured `.npz` traces, not a fresh live capture.

## Hardware Checks

- Functional FPGA check passed:
  - command: `python host/cw305_func_check.py --bitstream build/cw305_bnn_3.bit --kernels training/artifacts/model_3x3/layer1_kernels.npy --ksize 3 --freq 5e6`
  - result: all 784 outputs bit-exact against numpy golden.

## Paper Alignment

From `1803.05847v2.pdf`:

- Paper uses an FPGA line-buffer convolution accelerator.
- One output feature map cycle corresponds to one sliding convolution window.
- Power capture uses an oscilloscope at 2.5 GHz.
- Section 5 processing: low-pass filtering, DC restoration for an approximately 250 Hz
  high-pass measurement path, power alignment, RC curve fitting, trailing-power
  subtraction.
- Section 7 template attack uses 300 profiling images, 200 eval images, 9 kernels,
  group size 3, and delta 1.0.

Local deviations:

- CW-Lite samples at about 105 MS/s, not 2.5 GS/s. The paper's 60 MHz low-pass cutoff is
  above CW-Lite Nyquist, so `attack/power_extract.py` now clamps the cutoff instead of
  silently disabling the filter.
- Current HLS has `FANOUT=16` leakage amplification to make the signal observable with
  CW-Lite. Functionality remains bit-exact, but this is not strictly paper hardware.
  `build/run_hls.tcl` and `build/run_vivado.tcl` now support `FANOUT=1` builds for a
  paper-faithful bitstream.

## Pipeline Fixes

- `host/cw305_analog_capture.py`
  - stores capture manifest and per-file metadata (`sample_rate_hz`, `fpga_freq_hz`,
    `avg`, `gain_db`)
  - pre-packs kernels
  - supports `--start-index` and `--force`
- `attack/power_extract.py`
  - reads `sample_rate_hz` from trace metadata
  - clamps impossible low-pass cutoffs for CW-Lite captures
- `attack/run_on_hardware.py`
  - supports parallel extraction with `--jobs`
  - supports extraction cache
  - supports per-kernel S7 z-score normalization with `--normalize-rho`
- `attack/score_results.py`
  - scores saved hardware results with TensorFlow/golden MLP after capture processing.

## Big Runs

### Full Paper-Style Curve Fit

Command:

```powershell
python attack/run_on_hardware.py --ksize 3 --traces-dir ../host/traces_analog_live_500 `
  --n-profile 300 --n-eval 200 --frontend paper --jobs 8 --normalize-rho `
  --delta 1.0 --out results_live500_curvefit_norm
```

Scored with:

```bash
python attack/score_results.py --results results_live500_curvefit_norm
```

Results:

- `bg_pixel_acc_mean`: 0.6181
- `tm_pixel_dist_mean`: 31.9233
- `recog_acc_orig`: 0.975
- `recog_acc_background`: 0.120
- `recog_acc_template`: 0.140

### No Curve Fit Comparison

Command:

```powershell
python attack/run_on_hardware.py --ksize 3 --traces-dir ../host/traces_analog_live_500 `
  --n-profile 300 --n-eval 200 --frontend paper --no-curve-fit --jobs 8 `
  --normalize-rho --delta 1.0 --out results_live500_nofit_norm
```

Results:

- `bg_pixel_acc_mean`: 0.5474
- `tm_pixel_dist_mean`: 31.0327
- `recog_acc_orig`: 0.975
- `recog_acc_background`: 0.100
- `recog_acc_template`: 0.130

### Delta 0.5 Comparison

- `results_live500_nofit_norm_d05`
- `recog_acc_template`: 0.130

## Interpretation

The pipeline now uses live hardware traces and paper-sized 300/200 evaluation. The golden
classifier is working (`recog_acc_orig = 0.975`), but recovered-image recognition is far
below the paper target (`0.816` background / `0.898` template). Curve fitting improves
background pixel accuracy but does not recover template recognition.

Most likely blockers:

- CW-Lite analog bandwidth/sampling is much lower than the paper oscilloscope setup.
- Current HLS uses `FANOUT=16` leakage amplification, which changes the leakage model even
  though output function is bit-exact.
- The recovered template images have pixel distance around 31, much worse than the
  paper's reported 1.65 for Model 1.
