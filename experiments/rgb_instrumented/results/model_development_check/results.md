# Instrumented RGGB reconstruction evaluation

**Recorded acquisition sources: software_model. Status: ok.**

`software_model` denotes software-generated samples. `fpga` denotes records with the required hardware-validation flag; full supplied provenance is retained in metrics.json.

Inputs are explicit uint8 Bayer exports. Reconstruction receives only normalized mosaic samples and the fixed parameters listed below. Truth is used for validation, metrics, and display.

Scores use floating RGB in [0,1] before exported PNG rounding. Missing-channel scores exclude measured samples. Pooled PSNR is computed from pooled squared error; mean per-image PSNR is a separate arithmetic mean. Perfect reconstruction has infinite PSNR (JSON null with an explicit is_infinite flag).

Natural images and synthetic controls are summarized separately. These scenes are a small evaluation set and do not establish statistical generalization. Runtime covers reconstruction only.

Cases: 1; successful reconstructions: 4; recorded failures: 0.

## Fixed method parameters

```json
{
  "nearest": {},
  "bilinear": {},
  "smooth_ridge": {
    "ridge": 0.05,
    "rtol": 1e-07,
    "atol": 1e-10,
    "maxiter": 1000
  },
  "tv": {
    "weight": 0.05,
    "rtol": 1e-05,
    "atol": 1e-08,
    "maxiter": 2000,
    "check_every": 10
  }
}
```

`smooth_ridge` is graph-Laplacian inpainting with a ridge toward bilinear RGB. `tv` minimizes quadratic distance to bilinear RGB plus spatial isotropic TV. Both retain hard measured constraints. Neither parameter set is tuned using these evaluation images.

## Summaries by source, scene category, and resolution

| Source | Category | Resolution | Method | Passed / failed | Pooled RGB MSE | Pooled RGB PSNR dB | Mean image RGB PSNR dB | Pooled missing PSNR dB | Mean seconds |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| software_model | control | 32x32 | bilinear | 1 / 0 | 1.755423e-05 | 47.556 | 47.556 | 45.795 | 0.001 |
| software_model | control | 32x32 | nearest | 1 / 0 | 0.0003014866 | 35.207 | 35.207 | 33.446 | 0.001 |
| software_model | control | 32x32 | smooth_ridge | 1 / 0 | 2.591198e-05 | 45.865 | 45.865 | 44.104 | 0.009 |
| software_model | control | 32x32 | tv | 1 / 0 | 2.945361e-05 | 45.309 | 45.309 | 43.548 | 0.013 |

## Per-image results

| Case | Source | Method | RGB PSNR dB | Missing PSNR dB | R/G/B MAE | Measured max error | Seconds | Status |
|---|---|---|---:|---:|---|---:|---:|---|
| control_gradient_32 | software_model | nearest | 35.207 | 33.446 | 0.01618/0.00050/0.01525 | 0 | 0.001 | ok |
| control_gradient_32 | software_model | bilinear | 47.556 | 45.795 | 0.00141/0.00056/0.00132 | 0 | 0.001 | ok |
| control_gradient_32 | software_model | smooth_ridge | 45.865 | 44.104 | 0.00226/0.00056/0.00198 | 0 | 0.009 | ok |
| control_gradient_32 | software_model | tv | 45.309 | 43.548 | 0.00169/0.00027/0.00301 | 0 | 0.013 | ok |

## Comparison

The montage includes every successfully validated case at the largest image area present, with original RGB, grayscale Bayer samples, and each method. Captions report RGB PSNR; failed methods remain visibly marked.

![Reconstruction comparison](comparison.png)
