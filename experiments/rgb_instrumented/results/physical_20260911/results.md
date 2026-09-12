# Instrumented RGGB reconstruction evaluation

**Recorded acquisition sources: fpga. Status: ok.**

`software_model` denotes software-generated samples. `fpga` denotes records with the required hardware-validation flag; full supplied provenance is retained in metrics.json.

Inputs are explicit uint8 Bayer exports. Reconstruction receives only normalized mosaic samples and the fixed parameters listed below. Truth is used for validation, metrics, and display.

Scores use floating RGB in [0,1] before exported PNG rounding. Missing-channel scores exclude measured samples. Pooled PSNR is computed from pooled squared error; mean per-image PSNR is a separate arithmetic mean. Perfect reconstruction has infinite PSNR (JSON null with an explicit is_infinite flag).

Natural images and synthetic controls are summarized separately. These scenes are a small evaluation set and do not establish statistical generalization. Runtime covers reconstruction only.

Cases: 21; successful reconstructions: 84; recorded failures: 0.

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
| fpga | control | 128x128 | bilinear | 3 / 0 | 0.1273302 | 8.951 | 30.453 | 7.190 | 0.002 |
| fpga | control | 128x128 | nearest | 3 / 0 | 0.1176276 | 9.295 | 23.191 | 7.534 | 0.002 |
| fpga | control | 128x128 | smooth_ridge | 3 / 0 | 0.127501 | 8.945 | 29.846 | 7.184 | 0.016 |
| fpga | control | 128x128 | tv | 3 / 0 | 0.1272353 | 8.954 | 29.422 | 7.193 | 0.250 |
| fpga | control | 32x32 | bilinear | 3 / 0 | 0.1343646 | 8.717 | 22.470 | 6.956 | 0.000 |
| fpga | control | 32x32 | nearest | 3 / 0 | 0.1372533 | 8.625 | 17.094 | 6.864 | 0.000 |
| fpga | control | 32x32 | smooth_ridge | 3 / 0 | 0.1351832 | 8.691 | 21.783 | 6.930 | 0.003 |
| fpga | control | 32x32 | tv | 3 / 0 | 0.1339928 | 8.729 | 21.793 | 6.968 | 0.009 |
| fpga | control | 64x64 | bilinear | 3 / 0 | 0.1296673 | 8.872 | 26.473 | 7.111 | 0.001 |
| fpga | control | 64x64 | nearest | 3 / 0 | 0.1241565 | 9.060 | 20.150 | 7.299 | 0.001 |
| fpga | control | 64x64 | smooth_ridge | 3 / 0 | 0.1300164 | 8.860 | 25.847 | 7.099 | 0.004 |
| fpga | control | 64x64 | tv | 3 / 0 | 0.1294787 | 8.878 | 25.604 | 7.117 | 0.036 |
| fpga | natural | 128x128 | bilinear | 4 / 0 | 0.001636484 | 27.861 | 29.012 | 26.100 | 0.002 |
| fpga | natural | 128x128 | nearest | 4 / 0 | 0.004525658 | 23.443 | 24.706 | 21.682 | 0.002 |
| fpga | natural | 128x128 | smooth_ridge | 4 / 0 | 0.001884987 | 27.247 | 28.435 | 25.486 | 0.018 |
| fpga | natural | 128x128 | tv | 4 / 0 | 0.001651327 | 27.822 | 28.889 | 26.061 | 0.679 |
| fpga | natural | 32x32 | bilinear | 4 / 0 | 0.005619509 | 22.503 | 23.748 | 20.742 | 0.000 |
| fpga | natural | 32x32 | nearest | 4 / 0 | 0.01249718 | 19.032 | 20.170 | 17.271 | 0.000 |
| fpga | natural | 32x32 | smooth_ridge | 4 / 0 | 0.006116412 | 22.135 | 23.418 | 20.374 | 0.003 |
| fpga | natural | 32x32 | tv | 4 / 0 | 0.005710338 | 22.433 | 23.675 | 20.672 | 0.019 |
| fpga | natural | 64x64 | bilinear | 4 / 0 | 0.002969574 | 25.273 | 26.652 | 23.512 | 0.001 |
| fpga | natural | 64x64 | nearest | 4 / 0 | 0.007804883 | 21.076 | 22.470 | 19.315 | 0.001 |
| fpga | natural | 64x64 | smooth_ridge | 4 / 0 | 0.003408545 | 24.674 | 26.014 | 22.913 | 0.004 |
| fpga | natural | 64x64 | tv | 4 / 0 | 0.003005969 | 25.220 | 26.522 | 23.459 | 0.072 |

## Per-image results

| Case | Source | Method | RGB PSNR dB | Missing PSNR dB | R/G/B MAE | Measured max error | Seconds | Status |
|---|---|---|---:|---:|---|---:|---:|---|
| control_colorbars_128 | fpga | nearest | 17.392 | 15.631 | 0.00000/0.02344/0.03125 | 0 | 0.003 | ok |
| control_colorbars_128 | fpga | bilinear | 21.644 | 19.884 | 0.01953/0.01178/0.01562 | 0 | 0.002 | ok |
| control_colorbars_128 | fpga | smooth_ridge | 21.331 | 19.570 | 0.02861/0.01178/0.02288 | 0 | 0.022 | ok |
| control_colorbars_128 | fpga | tv | 21.869 | 20.108 | 0.01954/0.00943/0.01563 | 0 | 0.343 | ok |
| control_colorbars_32 | fpga | nearest | 11.372 | 9.611 | 0.00000/0.09375/0.12500 | 0 | 0.000 | ok |
| control_colorbars_32 | fpga | bilinear | 15.601 | 13.840 | 0.07812/0.04785/0.06250 | 0 | 0.000 | ok |
| control_colorbars_32 | fpga | smooth_ridge | 15.232 | 13.471 | 0.11459/0.04785/0.09143 | 0 | 0.003 | ok |
| control_colorbars_32 | fpga | tv | 15.821 | 14.060 | 0.07818/0.03871/0.06257 | 0 | 0.022 | ok |
| control_colorbars_64 | fpga | nearest | 14.382 | 12.621 | 0.00000/0.04688/0.06250 | 0 | 0.001 | ok |
| control_colorbars_64 | fpga | bilinear | 18.627 | 16.866 | 0.03906/0.02368/0.03125 | 0 | 0.001 | ok |
| control_colorbars_64 | fpga | smooth_ridge | 18.307 | 16.547 | 0.05737/0.02368/0.04589 | 0 | 0.004 | ok |
| control_colorbars_64 | fpga | tv | 18.850 | 17.089 | 0.03908/0.01903/0.03126 | 0 | 0.060 | ok |
| control_gradient_128 | fpga | nearest | 47.427 | 45.666 | 0.00392/0.00003/0.00391 | 0 | 0.002 | ok |
| control_gradient_128 | fpga | bilinear | 65.456 | 63.695 | 0.00008/0.00003/0.00008 | 0 | 0.002 | ok |
| control_gradient_128 | fpga | smooth_ridge | 63.948 | 62.187 | 0.00013/0.00003/0.00014 | 0 | 0.015 | ok |
| control_gradient_128 | fpga | tv | 62.138 | 60.377 | 0.00013/0.00001/0.00028 | 0 | 0.393 | ok |
| control_gradient_32 | fpga | nearest | 35.207 | 33.446 | 0.01618/0.00050/0.01525 | 0 | 0.000 | ok |
| control_gradient_32 | fpga | bilinear | 47.556 | 45.795 | 0.00141/0.00056/0.00132 | 0 | 0.000 | ok |
| control_gradient_32 | fpga | smooth_ridge | 45.865 | 44.104 | 0.00226/0.00056/0.00198 | 0 | 0.003 | ok |
| control_gradient_32 | fpga | tv | 45.309 | 43.548 | 0.00169/0.00027/0.00301 | 0 | 0.005 | ok |
| control_gradient_64 | fpga | nearest | 41.332 | 39.571 | 0.00797/0.00012/0.00772 | 0 | 0.001 | ok |
| control_gradient_64 | fpga | bilinear | 56.537 | 54.776 | 0.00034/0.00013/0.00032 | 0 | 0.001 | ok |
| control_gradient_64 | fpga | smooth_ridge | 54.978 | 53.217 | 0.00057/0.00013/0.00052 | 0 | 0.004 | ok |
| control_gradient_64 | fpga | tv | 53.706 | 51.945 | 0.00045/0.00006/0.00093 | 0 | 0.044 | ok |
| control_nyquist_128 | fpga | nearest | 4.754 | 2.993 | 0.50000/0.00391/0.50000 | 0 | 0.002 | ok |
| control_nyquist_128 | fpga | bilinear | 4.258 | 2.497 | 0.50000/0.25000/0.50000 | 0 | 0.002 | ok |
| control_nyquist_128 | fpga | smooth_ridge | 4.258 | 2.497 | 0.50000/0.25000/0.50000 | 0 | 0.010 | ok |
| control_nyquist_128 | fpga | tv | 4.257 | 2.496 | 0.50000/0.25000/0.50000 | 0 | 0.014 | ok |
| control_nyquist_32 | fpga | nearest | 4.704 | 2.943 | 0.50000/0.01562/0.50000 | 0 | 0.000 | ok |
| control_nyquist_32 | fpga | bilinear | 4.253 | 2.492 | 0.50000/0.25000/0.50000 | 0 | 0.000 | ok |
| control_nyquist_32 | fpga | smooth_ridge | 4.253 | 2.492 | 0.50000/0.25000/0.50000 | 0 | 0.002 | ok |
| control_nyquist_32 | fpga | tv | 4.251 | 2.490 | 0.50000/0.25000/0.50000 | 0 | 0.001 | ok |
| control_nyquist_64 | fpga | nearest | 4.737 | 2.977 | 0.50000/0.00781/0.50000 | 0 | 0.001 | ok |
| control_nyquist_64 | fpga | bilinear | 4.256 | 2.496 | 0.50000/0.25000/0.50000 | 0 | 0.001 | ok |
| control_nyquist_64 | fpga | smooth_ridge | 4.256 | 2.496 | 0.50000/0.25000/0.50000 | 0 | 0.003 | ok |
| control_nyquist_64 | fpga | tv | 4.255 | 2.494 | 0.50000/0.25000/0.50000 | 0 | 0.004 | ok |
| natural_astronaut_128 | fpga | nearest | 20.000 | 18.239 | 0.04668/0.03321/0.04984 | 0 | 0.002 | ok |
| natural_astronaut_128 | fpga | bilinear | 24.472 | 22.711 | 0.03126/0.01766/0.03193 | 0 | 0.002 | ok |
| natural_astronaut_128 | fpga | smooth_ridge | 23.852 | 22.091 | 0.03734/0.01766/0.03844 | 0 | 0.017 | ok |
| natural_astronaut_128 | fpga | tv | 24.483 | 22.723 | 0.03290/0.01622/0.03321 | 0 | 0.367 | ok |
| natural_astronaut_32 | fpga | nearest | 15.900 | 14.139 | 0.10051/0.06504/0.09411 | 0 | 0.000 | ok |
| natural_astronaut_32 | fpga | bilinear | 19.416 | 17.655 | 0.06303/0.03649/0.07412 | 0 | 0.000 | ok |
| natural_astronaut_32 | fpga | smooth_ridge | 18.990 | 17.229 | 0.07238/0.03649/0.08332 | 0 | 0.003 | ok |
| natural_astronaut_32 | fpga | tv | 19.326 | 17.565 | 0.06489/0.03496/0.07632 | 0 | 0.012 | ok |
| natural_astronaut_64 | fpga | nearest | 17.554 | 15.793 | 0.06698/0.04923/0.07397 | 0 | 0.001 | ok |
| natural_astronaut_64 | fpga | bilinear | 21.753 | 19.993 | 0.04593/0.02605/0.05043 | 0 | 0.001 | ok |
| natural_astronaut_64 | fpga | smooth_ridge | 21.237 | 19.476 | 0.05387/0.02605/0.05856 | 0 | 0.005 | ok |
| natural_astronaut_64 | fpga | tv | 21.744 | 19.984 | 0.04784/0.02455/0.05188 | 0 | 0.051 | ok |
| natural_chelsea_128 | fpga | nearest | 26.610 | 24.849 | 0.03025/0.01800/0.02917 | 0 | 0.002 | ok |
| natural_chelsea_128 | fpga | bilinear | 31.077 | 29.316 | 0.01877/0.01013/0.01776 | 0 | 0.002 | ok |
| natural_chelsea_128 | fpga | smooth_ridge | 30.421 | 28.660 | 0.02112/0.01013/0.02015 | 0 | 0.019 | ok |
| natural_chelsea_128 | fpga | tv | 30.680 | 28.919 | 0.02033/0.01008/0.01957 | 0 | 0.313 | ok |
| natural_chelsea_32 | fpga | nearest | 22.414 | 20.653 | 0.04624/0.03308/0.04927 | 0 | 0.000 | ok |
| natural_chelsea_32 | fpga | bilinear | 25.936 | 24.175 | 0.03412/0.01922/0.03404 | 0 | 0.000 | ok |
| natural_chelsea_32 | fpga | smooth_ridge | 25.692 | 23.931 | 0.03638/0.01922/0.03667 | 0 | 0.003 | ok |
| natural_chelsea_32 | fpga | tv | 25.792 | 24.031 | 0.03625/0.01937/0.03552 | 0 | 0.013 | ok |
| natural_chelsea_64 | fpga | nearest | 24.124 | 22.363 | 0.03913/0.02398/0.03893 | 0 | 0.001 | ok |
| natural_chelsea_64 | fpga | bilinear | 28.217 | 26.456 | 0.02549/0.01333/0.02435 | 0 | 0.001 | ok |
| natural_chelsea_64 | fpga | smooth_ridge | 27.651 | 25.890 | 0.02843/0.01333/0.02786 | 0 | 0.004 | ok |
| natural_chelsea_64 | fpga | tv | 27.863 | 26.102 | 0.02762/0.01332/0.02679 | 0 | 0.045 | ok |
| natural_coffee_128 | fpga | nearest | 23.355 | 21.594 | 0.03055/0.01786/0.02815 | 0 | 0.002 | ok |
| natural_coffee_128 | fpga | bilinear | 27.808 | 26.047 | 0.01825/0.01175/0.01872 | 0 | 0.002 | ok |
| natural_coffee_128 | fpga | smooth_ridge | 27.154 | 25.393 | 0.02160/0.01175/0.02219 | 0 | 0.019 | ok |
| natural_coffee_128 | fpga | tv | 27.886 | 26.125 | 0.01861/0.01092/0.01877 | 0 | 0.742 | ok |
| natural_coffee_32 | fpga | nearest | 18.366 | 16.605 | 0.06928/0.03602/0.06093 | 0 | 0.000 | ok |
| natural_coffee_32 | fpga | bilinear | 21.600 | 19.839 | 0.04885/0.02466/0.04469 | 0 | 0.000 | ok |
| natural_coffee_32 | fpga | smooth_ridge | 21.273 | 19.512 | 0.05479/0.02466/0.04861 | 0 | 0.003 | ok |
| natural_coffee_32 | fpga | tv | 21.585 | 19.824 | 0.05059/0.02386/0.04489 | 0 | 0.024 | ok |
| natural_coffee_64 | fpga | nearest | 21.043 | 19.282 | 0.04534/0.02518/0.04024 | 0 | 0.001 | ok |
| natural_coffee_64 | fpga | bilinear | 25.287 | 23.526 | 0.02827/0.01678/0.02784 | 0 | 0.001 | ok |
| natural_coffee_64 | fpga | smooth_ridge | 24.517 | 22.756 | 0.03389/0.01678/0.03308 | 0 | 0.004 | ok |
| natural_coffee_64 | fpga | tv | 25.334 | 23.573 | 0.02933/0.01553/0.02820 | 0 | 0.086 | ok |
| natural_rocket_128 | fpga | nearest | 28.859 | 27.098 | 0.01298/0.00722/0.01040 | 0 | 0.002 | ok |
| natural_rocket_128 | fpga | bilinear | 32.690 | 30.929 | 0.00863/0.00386/0.00775 | 0 | 0.002 | ok |
| natural_rocket_128 | fpga | smooth_ridge | 32.312 | 30.551 | 0.00974/0.00386/0.00841 | 0 | 0.017 | ok |
| natural_rocket_128 | fpga | tv | 32.507 | 30.746 | 0.00907/0.00362/0.00810 | 0 | 1.297 | ok |
| natural_rocket_32 | fpga | nearest | 23.999 | 22.238 | 0.02876/0.01556/0.01587 | 0 | 0.000 | ok |
| natural_rocket_32 | fpga | bilinear | 28.039 | 26.278 | 0.01858/0.00700/0.01113 | 0 | 0.000 | ok |
| natural_rocket_32 | fpga | smooth_ridge | 27.715 | 25.954 | 0.02086/0.00700/0.01219 | 0 | 0.004 | ok |
| natural_rocket_32 | fpga | tv | 27.998 | 26.237 | 0.01945/0.00601/0.01211 | 0 | 0.028 | ok |
| natural_rocket_64 | fpga | nearest | 27.161 | 25.400 | 0.01693/0.01001/0.01227 | 0 | 0.001 | ok |
| natural_rocket_64 | fpga | bilinear | 31.353 | 29.592 | 0.01146/0.00484/0.00859 | 0 | 0.001 | ok |
| natural_rocket_64 | fpga | smooth_ridge | 30.650 | 28.890 | 0.01321/0.00484/0.00938 | 0 | 0.004 | ok |
| natural_rocket_64 | fpga | tv | 31.147 | 29.386 | 0.01216/0.00439/0.00917 | 0 | 0.105 | ok |

## Comparison

The montage includes every successfully validated case at the largest image area present, with original RGB, grayscale Bayer samples, and each method. Captions report RGB PSNR; failed methods remain visibly marked.

![Reconstruction comparison](comparison.png)
