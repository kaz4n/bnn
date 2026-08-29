# Hardware Run 2026-08-27: Re-running the New Method, Faithfulness Analysis, and Power2Picture Plan

Everything below comes from traces captured on the bench during this session.
No result reuses the trace directories behind the existing paper
(`host/traces_leakage_60_l63`, `host/traces_active_template_fresh2`).

## 1. Bench state

```text
ChipWhisperer-Lite   sn 50203220415447303030313238323035
CW305 Artix FPGA     sn 5020322030384a583330343234303030
bitstream            build/cw305_leakage_d8_l63.bit
FPGA clock           5 MHz, ADC extclk_x4 (4 samples/clock), gain 40 dB
core                 3x3 valid binary conv, DWELL=8, LEAK_LANES=63
```

Functional check before capture (`host/cw305_leakage_func_check.py`, image 100,
one-hot kernel 0):

```text
hw range=-5..9 golden range=-5..9
PASS: 676 outputs bit-exact
```

## 2. Captures taken this session

| id | directory | content | purpose |
|---|---|---|---|
| A | `host/traces_fresh_onehot_100_159` | 60 images x 9 one-hot kernels x 5 repeats | re-run of the report method on new images |
| B | `host/traces_fresh_trained_100_159` | 60 images x 9 **trained** conv kernels x 5 repeats | drop the one-hot probe crutch |
| C | `host/traces_p2p_9k` | 9000 images x 1 trained kernel x 1 shot | Power2Picture generator dataset |
| D | `host/traces_wei_scale_500` | 500 images x 9 one-hot kernels x 3 repeats | Wei et al. dataset scale (300 profile / 200 attack) |

Attack results live in `attack/fresh_run/` (captures A and B),
`attack/results_p2p_clockabs/` (capture C) and `attack/wei_scale/` (capture D).
The 200-image attack was split into three partitions
(`wei_scale/results_200_p{0,1,2}`) and merged into
`wei_scale/results_200_merged` for scoring.

Capture C uses a new script, `host/cw305_p2p_capture.py`: one single-shot trace
per image, int16 shards, 35 img/s, 231 MB for 9000 images. It also runs the
bit-exact functional check before the bulk loop.

## 3. What ran, and what came out

All routes are scored with one metric set (`attack/score_all.py`): pixel
accuracy, foreground F1/IoU, MSSIM (11x11, as in Power2Picture), Wei's
pixel-level distance on the 0-255 scale, and recognition accuracy through the
project's own BinaryNet MLP (`attack/recognize_numpy.py`, a NumPy re-write of
the TensorFlow golden model: 98.35% on grayscale MNIST, 98.20% binarized).

```text
| metric | template_trained_kernels | template_onehot_wei_scale_200 | template_onehot_fresh30 | template_onehot_percycle | template_onehot_crosssession | p2p_generator_500 | background_detection_onehot_k0 | background_detection_mean9 |
|---|---|---|---|---|---|---|---|---|
| N | 30 | 200 | 30 | 30 | 30 | 500 | 30 | 30 |
| pixel acc | 0.9676 | 0.9431 | 0.9468 | 0.9490 | 0.9406 | 0.9721 | 0.8747 | 0.8754 |
| all-zero baseline | 0.8785 | 0.8752 | 0.8785 | 0.8785 | 0.8785 | 0.8713 | 0.8785 | 0.8785 |
| fg F1 | 0.8509 | 0.7779 | 0.7670 | 0.7700 | 0.7428 | 0.8909 | 0.4980 | 0.4859 |
| fg IoU | 0.7405 | 0.6365 | 0.6221 | 0.6261 | 0.5908 | 0.8033 | 0.3316 | 0.3209 |
| MSSIM | 0.824 | 0.704 | 0.725 | 0.751 | 0.693 | 0.803 | 0.341 | 0.382 |
| pixel dist (0-255) | 8.26 | 14.52 | 13.56 | 13.01 | 15.15 | 8.82 | 31.95 | 31.78 |
| recognition | 0.9333 | 0.9450 | 1.0000 | 1.0000 | 1.0000 | 0.9460 | 0.9000 | 0.8667 |
| recognition (orig) | 1.0000 | 0.9850 | 1.0000 | 1.0000 | 1.0000 | 0.9740 | 1.0000 | 1.0000 |
```

Read it as follows.

* **Trained kernels are the best template route.** Capture B (the model's own
  kernels) beats the one-hot probes on every pixel metric. With one-hot kernels
  the XNOR match count is squeezed into 7-9, so patches overlap in leakage
  level; mixed +-1 trained kernels spread the match-count distribution. The
  recognition column inverts (93.3% vs 100%), but at n=30 that is one or two
  images, so the routes should be ranked on the pixel metrics.
* **Templates port across capture sessions.** `template_onehot_crosssession`
  uses the *previous* session's template (`attack/active_template_hw/profile_template.npz`)
  against traces captured tonight after re-programming the FPGA and re-arming
  the scope: F1 0.743 vs 0.767 same-session, recognition still 100%.
* **Per-cycle features help slightly.** `cycle_abs` (8 values per window per
  kernel) gives F1 0.770 vs 0.767 and MSSIM 0.751 vs 0.725.
* **The Wei-scale run matches their protocol size.** 300 profiling images and
  200 held-out attack images, the same split as the reference paper:
  94.31% pixel accuracy, F1 0.778, recognition **94.5%** against their reported
  89.8% for the 3x3 power-template attack. More profiling data helps (F1 0.778
  at 300 profile images vs 0.767 at 30).
* **The generator is the strongest route overall** and it uses the cheapest
  measurement: one single-shot trace per image with the trained kernel.
* **Background detection is weak here, and that is the informative part.** Its
  pixel accuracy (0.8747) is *below* the all-zero baseline (0.8785); only F1,
  MSSIM and recognition carry signal. See section 4.2 item 1.

Two comparisons that must not be made:

* The pixel-distance row measures a 1-bit silhouette. Wei's 1.65 is on 8-bit
  grayscale pixel values. Different quantity - not a worse result.
* Recognition accuracy *is* comparable across all three works, because all
  three feed recovered images to an independent classifier.

Reference numbers: Wei et al. report 86.2% pixel accuracy / 81.6% recognition
for background detection and 89.8% recognition for the 3x3 power-template
attack. Power2Picture report 96.4% recognition and MSSIM 0.62 for MNIST on the
PYNQ-Z1.

## 4. Faithfulness gap vs. Wei et al. (1803.05847v2)

### 4.1 What the current method already matches

* First convolution layer of a binary CNN accelerator on an FPGA is the target.
* Active/profiled power template: profile images -> `{patch bits, power vector}`,
  grouped candidate query, overlap-consistency selection (their Algorithm 2).
* Held-out attack images, trace-only attack bundle (no images or labels).
* Recognition accuracy measured by an independent classifier, like their 99.2%
  golden reference.

### 4.2 The five real gaps, in order of how much they distort the claim

1. **Leakage mechanism.** Wei's leakage is a line buffer shifting *pixel values*
   through registers, so the per-cycle power is a Hamming-weight/-distance
   function of the pixels themselves; background pixels produce almost no
   toggling. The CW305 core instead toggles a retained bank with
   `XNOR(window, kernel)`, so power is a function of the *match count*, not of
   the pixel values. This is why background detection behaves differently here
   (Section 5) and why the leak bank has to be called an amplifier rather than a
   faithful copy of the victim datapath.
2. **Input precision.** Wei attacks 8-bit grayscale MNIST and recovers pixel
   *values* (their pixel-level distance 1.65 on the 0-255 scale). The CW305 core
   takes a 1-bit binarized image, so the attack recovers a silhouette, and a
   pixel-distance comparison against their 1.65 is not apples-to-apples.
3. **Probe kernels.** The report's method uses nine one-hot probe kernels. Wei's
   adversary uses the deployed model's own kernels (9 of the 64 trained kernels).
   **This gap is now closed** - capture B ran the trained kernels and scored
   *better* than the one-hot probes (Section 3).
4. **Power extraction.** Wei extract per clock cycle: low-pass filter, template
   correlation for cycle alignment, RC curve fitting, trailing-power subtraction.
   The report collapses each 32-sample window to one `mean_abs` scalar. A
   `cycle_abs` feature mode (8 features per window per kernel) is now implemented
   in `attack/leakage_first_reconstruct.py` and selectable in the active-template
   attack; `attack/power_extract.py` already contains the full paper-style
   analog front end from the earlier work and can be wired to the leakage core.
5. **Dataset size.** Wei use 300 profiling and 200 attack images. The report used
   30/30. Capture D closes this too (Section 3).

### 4.3 Gaps that hardware forces, and cannot be closed on this bench

* Measurement bandwidth: 4 ADC samples per FPGA clock at 5 MHz versus their
  2.5 GS/s oscilloscope. The dwell-8 window is the workaround.
* The retained leak bank (567 data-dependent FFs) exists because the natural
  line-buffer leakage is below the CW-Lite noise floor on this board.

### 4.4 Concrete change list to go further (gateware work, not done this session)

1. Replace the XNOR leak bank with a **line-buffer shift register** holding the
   pixel row: 28 registers of the pixel word, shifted every cycle. Power then
   rides the Hamming distance between consecutive pixels, which is Wei's actual
   mechanism, and background detection recovers its natural "low power =
   background" polarity.
2. Widen the pixel word from 1 bit to 8 bits so that the recovered quantity is a
   pixel *value*; keep the binarization inside the MAC so the network stays a
   BNN. Then the pixel-level distance is directly comparable to their 1.65.
3. Stream all 64 trained kernels in one run rather than one kernel per capture,
   so one trace covers a whole conv layer, as their accelerator does.
4. Feed the captured traces through `attack/power_extract.py` in `paper` mode
   (align, RC fit, trailing subtraction) instead of `mean_abs`.

## 5. Background detection (Wei Section 6) - newly implemented

`attack/background_detect.py` is the passive half of the reference paper: no
power template, no per-pixel profiling, one scalar threshold taken from the
profiling images, background markers unioned over overlapping windows.

Two variants are provided: `--combine single` (one probe kernel, as in the
paper) and `--combine mean` (mean over the nine kernels; for one-hot kernels the
mean match count is `8 - (7/9)s` in the number of foreground pixels `s`).

The measured behaviour matches the paper's *numbers* but exposes the mechanism
gap of Section 4.2: with an XNOR-match leak, an all-background window is not the
lowest-power window, so the detector is weaker than the same idea on a real line
buffer would be.

## 6. Power2Picture (Huegle et al., FCCM 2023) - pilot implemented

The proposal splits into two independent axes, and they should not be conflated:

* **Axis A - channel.** The paper measures with on-chip TDC sensors on a
  multi-tenant FPGA (remote attacker, no physical access). This bench has an
  external CW-Lite shunt measurement on a single-tenant CW305. The repo already
  has RO-counter sensor gateware (`cw305/ro_counter_sensor_v2.v`,
  `host/ro_capture.py`) and a prior finding that the on-chip channel is about 2x
  the external one but dilution-limited (correlation plateau ~0.33).
* **Axis B - attack model.** Hand-built template decoder versus a learned
  generative CNN. This axis is testable *today* on the external channel, which is
  what this session did.

Implemented in `attack/p2p_generator.py`, following the paper:

* Generator: `Linear 128 -> Linear 128 -> Linear 12544 -> reshape (256,7,7) ->
  ConvT 128 (5,5) s1 p2 -> ConvT 64 (4,4) s2 p1 -> ConvT 1 (4,4) s2 p1`.
* Loss `gradmse = MSE + MSE(dx) + MSE(dy)`; Adam lr 1e-3; batch 256.
* Metrics: recognition accuracy, MSSIM, pixel-level distance - their exact set.
* Input feature modes: `raw` (22144 ADC samples), `clock_abs` (5408, one per FPGA
  clock), `window_abs` (676, one per convolution window).

Why this route is *cheaper and more faithful* than the template route: the
generator needs no probe kernels and no repeats, so one image costs one trace
instead of 45. That is how 9000 images fit in ~4 minutes of board time.

### 6.1 Roadmap to a full Power2Picture reproduction

1. **Done:** external-channel pilot, 9000 real traces, paper generator and loss,
   paper metric set.
2. **Grayscale victim:** the generator can output grayscale, but the CW305 core
   currently consumes a 1-bit image, so the target must carry 8-bit pixels first
   (same gateware change as Section 4.4 item 2).
3. **On-chip sensor channel:** re-capture the same 9000-image protocol through
   the RO/TDC sensor instead of the CW-Lite shunt. Only the capture script
   changes; the training script consumes the same shard format.
4. **Portability experiments from their Table II:** re-place-and-route the
   bitstream and attack with a generator trained on the previous build; if a
   second CW305 is available, cross-board; temperature variation needs a chamber
   and is out of scope here.
5. **Second dataset:** Fashion-MNIST, same pipeline, to show it is not
   MNIST-specific.
6. **Multi-layer victim:** their victim is a full quantized LeNet through FINN.
   The CW305 core is a single conv layer; a FINN-style multi-layer accelerator is
   a separate build effort.

## 7. Code added or changed this session

| file | what |
|---|---|
| `host/cw305_p2p_capture.py` | new: bulk single-shot capture, int16 shards, functional check |
| `attack/p2p_generator.py` | new: Power2Picture generator, gradmse loss, MSSIM/pixel-distance metrics |
| `attack/background_detect.py` | new: Wei Section 6 background detection |
| `attack/recognize_numpy.py` | new: NumPy BinaryNet MLP, removes the TensorFlow dependency for recognition accuracy |
| `attack/score_all.py`, `attack/results_table.py`, `attack/make_montage.py` | new: one metric set and one table/figure path for every route |
| `attack/leakage_first_reconstruct.py` | added `cycle_abs` per-clock-cycle feature mode |
| `attack/active_power_template_cw305.py` | added `cycle_abs` feature option; added `greedy_reconstruct_fast` (bit-identical to the original, 14.6x faster) with `--slow-reconstruct` to fall back |

The fast reconstruction was verified against the original implementation on a
real attack window set: `image identical: True  codes identical: True`.
