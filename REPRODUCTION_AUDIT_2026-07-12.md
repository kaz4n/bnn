# Reproduction audit and revised methodology

Date: 2026-07-12

Target paper: Wei et al., "I Know What You See: Power Side-Channel Attack on
Convolutional Neural Network Accelerators" (ACSAC 2018, arXiv:1803.05847v2).

Current platform: CW305-A100 (Artix-7 XC7A100T) measured with a
ChipWhisperer-Lite (CW1173).

## Executive conclusion

The project has a credible bit-exact convolution path and a working CW305/CW-Lite
capture connection. It does not yet have evidence for either of these stronger claims:

1. a faithful physical reproduction of the paper; or
2. a fundamental CW305/CW-Lite measurement floor.

The existing hardware recovery failure is real, but its cause is not isolated. The
principal run used a bitstream that predates the current kernel-dependent MAC fanout,
the acquisition averaged asynchronous traces before alignment, and one later dataset
captured 784 physical clocks even though the first-to-last output span is 838 clocks.
The paper-specific 250 Hz high-pass inverse and RC model were also reused without
characterizing the CW305 X4 plus CW-Lite channel.

The correct current conclusion is:

> The functional accelerator and capture plumbing work. Existing measured traces do
> not recover images, but the repository has not yet separated DUT leakage,
> acquisition timing, analog-channel response, preprocessing, and attack fidelity.

## What the paper actually establishes

- DUT: Spartan-6 LX75 on SAKURA-G.
- Measurement: 1 ohm supply shunt, SAKURA-G amplified AC signal, Tektronix
  MDO3034 sampled at 2.5 GS/s.
- Front end: 60 MHz low-pass, inversion of a board-specific approximately 250 Hz
  high-pass response, manual cycle-template alignment, RC curve fitting, and trailing
  energy subtraction.
- Accelerator: line-buffer convolution with one input/output operation per cycle.
- Models: MNIST 28x28x1, 64 first-layer kernels, K=3 and K=5 variants.
- Passive experiment: 500 randomly selected test images; Eq. 4 histogram threshold.
- Active experiment: 300 profiling images, 200 evaluation images, 9 kernels, three
  groups of three, delta=1.0.
- Reported K=3 targets: 86.2% background pixel accuracy, 81.6% background-image
  recognition, 89.8% template-image recognition, and 1.65 mean pixel distance.

The paper does not report the DUT clock, padding convention, trigger definition,
averaging count, filter order, fit solver/bounds, tail duration, exact random split,
weights, or training recipe. Figure 4 visually suggests hundreds rather than
"thousands" of samples per clock, but the clock is not stated.

The paper's statement that the convolution unit is more than 80% of total power refers
to the simulated line-buffer block (0.57 mW of 0.67 mW for K=3), not the entire FPGA or
board. It cannot be transferred to whole-chip CW305 VCCINT power.

## Hardware and measurement differences

| Dimension | Paper | Current platform | Required treatment |
|---|---|---|---|
| FPGA | Spartan-6 LX75 | Artix-7 XC7A100T | Re-profile leakage; do not transfer a template |
| Board | SAKURA-G SCA board | CW305-A100 | Identify board revision, shunt, decoupling, and supply state |
| Analog path | 1 ohm shunt plus SAKURA amplifier | CW305 X4 20 dB VCCINT low-side amplifier plus CW-Lite input | Measure the combined transfer function |
| ADC | Tek MDO3034, 2.5 GS/s | 10-bit CW-Lite, 105 MS/s | Use synchronous x4 as primary; async only with per-repeat alignment |
| Input coupling | Paper amplifier is AC-only | CW-Lite input is AC-coupled | Recover cycle energy with calibrated system identification |
| Clocking | Unreported | CW305 CDCE906, CW-Lite x1/x4 or independent ADC clock | Record actual frequencies and lock state per capture |
| Threat model | Passive trace or active pre-profiling | Image and kernel replayed and averaged 20-100 times | Report single-shot primary and averaging curves separately |
| Kernel schedule | Model-resident kernels | Host loads one kernel per invocation | Store/schedule nine kernels in one inference for the faithful track |

Official NewAE documentation states that CW305 X4 is a 20 dB amplified low-side
VCCINT measurement and that the CW-Lite is a 10-bit, 105 MS/s, AC-coupled capture
device supporting x1/x4 synchronous sampling. Those facts support this platform, but
they do not imply that four samples per cycle are already clean cycle power. Clock
synchrony removes drift; it does not remove PDN or analog-channel convolution.

## Critical repository findings

### 1. Captured bitstream and current source do not match

The principal F16 bitstream is dated 2026-07-04. The current
`hls/bnn_conv1.cpp` is dated 2026-07-08.

The HLS-preserved source used for the old F16 build duplicates the line buffer and
window 16 times, while its MAC reads only `win[0]`. It therefore amplifies mostly
kernel-independent pixel-shift activity, not the across-kernel signal required by
Section 7. The current source instead fans out kernel-dependent MAC lanes, but there is
no matching HLS/Vivado artifact or hardware capture.

Consequence: the statement that a 16x kernel-dependent amplifier failed, proving a
measurement floor, is unsupported. Rebuild F1/F2/F4/F8/F16 from exact hashed sources
and verify retained lanes in synthesis and implementation reports.

### 2. Trigger and raster timing are not what the documentation claims

The current HLS scans an extended raster:

```
EW = 28 + K - 1
physical clocks = EW * EW
```

For K=3, EW=30. The accelerator writes 28 outputs, has two no-write clocks, and
repeats this for every row. The wrapper drives `tio_trigger` from each output write, so
it is a pulse train rather than a continuous 784-clock window.

For a capture triggered by the first output, the inclusive physical span to the last
output is:

```
span = (LINE - 1) * (EW + 1) + 1
```

- K=3: 838 clocks.
- K=5: 892 clocks.

`host/traces_physical_v1` records `capture_cycles="output"`, `n_cycles=784`, and a
first-output trigger. It is therefore row-drifting and truncated; its associated
results are invalid. `host/traces_analog_trigfix_500` retains 900 cycles with offset
62 and is the best existing exploratory hardware dataset.

Longer term, the faithful hardware should expose a continuous `measure_active` signal
for exactly 784 accepted real input pixels and a separate `output_valid` marker.
Padding/flush cycles should not be inserted into the measured input stream.

### 3. Asynchronous repeats are averaged before alignment

`host/cw305_analog_capture.py` pointwise-adds raw traces and saves only their mean.
With independent ADC and DUT clocks, repeats can differ in fractional phase and period.
This destroys the sub-cycle transient before Section 5 processing can recover it.

Store `[kernel, repeat, sample]`. Estimate one constrained phase/period model for each
repeat, resample to a common grid, extract features, then combine with a robust mean or
median. Never average unaligned asynchronous samples.

### 4. Current Section 7 code is an adapted attack

- The paper's written group distance is L1 across scalar kernel powers. Current KDTree
  search uses squared Euclidean distance.
- The paper uses strict intersection. Current code falls back to the union when the
  intersection is empty.
- Paper Algorithm 2 tries every candidate from a randomly selected seed set and
  dynamically selects the unprocessed cycle with maximum overlap. Current code uses six
  fixed seed cycles and rotated scan order.
- Current power vectors are z-scored and the CLI defaults to delta=0.1. The paper uses
  raw extracted power and delta=1.0.
- Current passive recovery defaults to Otsu; the paper uses Eq. 4.

These adaptations can be valuable, but an exact baseline and an adapted track must be
reported separately.

### 5. Stored results do not beat trivial baselines

On evaluation images 300-499:

- all-background pixel accuracy: 0.8192;
- all-black mean pixel distance: 30.973.

The corrected trigfix runs report:

- background pixel accuracy: 0.586 (no fit) or 0.635 (curve fit), both worse than the
  all-background baseline;
- template distance: 31.077 or 31.777, both worse than all black;
- template recognition: 0.13, but the no-fit result predicts digit 1 for all 200
  images and the curve-fit result predicts digit 1 for 194/200. The evaluation set
  contains 26 digit-1 images, so 0.13 is class prevalence, not recovered information.

The present DL-SCA result (0.742 background-pixel accuracy) is also below the 0.8192
all-background baseline.

### 6. Simulation is internally consistent, not physically validated

The simulator and attack share the same ideal 784-cycle leakage model. It omits the
extended raster, row bubbles, control/routing/BRAM activity, analog transfer function,
and the exact old bitstream. Its K=3 artifact reports 0.93 recognition, but mean pixel
distance is 28.32 rather than the paper's 1.65. No current 5x5 summary artifact exists.

Use simulation for unit tests and sensitivity studies, not as evidence of physical
paper reproduction.

### 7. Other correctness and reproducibility defects

- Status returns 0xAB when busy and 0x77 when idle; both have bit zero set although host
  code treats bit zero as the busy flag.
- The old F16 HLS report has 5,630-cycle latency, including 3,376 initialization clocks,
  not the documented 1,593 cycles. Pre-trigger switching can bleed into the capture.
- The on-chip RO block is not a frequency counter; it synchronizes taps and counts
  observed transitions (0-16), with severe aliasing. It is disabled in normal builds.
- The Vivado script does not add `cw305/cw305_bnn.xdc` and can silently select a stale
  HLS directory.
- Data-dependent parity drives a physical LED, adding non-paper IO switching.
- A new capture manifest can overwrite the description of existing skipped files.
- Extraction cache keys omit trace content, code, and bitstream hashes.
- Relative paths can silently fall back to random kernels.
- The 3x3 attacked BNN is 93.01% accurate versus the paper's 99.42%; the golden MLP is
  98.84% versus 99.2%.
- Test data are used for training validation and later attack evaluation; seeds and
  dependency versions are not fixed.

## Revised experiment: two explicit tracks

### Track A - paper baseline

- FANOUT=1.
- Continuous, deterministic one-real-pixel-per-clock line-buffer schedule.
- Nine fixed kernels run in model order during one inference.
- Single-shot attack traces are the primary result.
- Paper Eq. 4, raw feature units, L1 group distance, delta=1.0, strict intersection,
  and literal Algorithm 2.
- Paper metrics and Table 2 candidate statistics.

### Track B - CW305-adapted attack

- Controlled FANOUT variants, clearly labelled non-paper hardware.
- Calibrated channel equalization, waveform features, whitening/Mahalanobis distance,
  Otsu, PCA or supervised profiled models.
- Replay/averaging and ADC phase sweeps allowed only when explicitly reported.
- Every adaptation ablated against Track A.

## Offline phase (hardware disconnected)

1. Freeze an immutable source revision and record the dirty-diff hash.
2. Fix status, reset/start handling, trigger duration, full-span mapping, and raw-repeat
   storage.
3. Make HLS/Vivado fail closed on missing or stale inputs; add the custom XDC.
4. Add build-ID registers and SHA-256 manifests for source, RTL, bitstream, kernels, and
   images.
5. Rebuild current F1 and F16, then K=5, and verify:
   - C simulation against numpy for multiple images and all 64 kernels;
   - II=1 for the measured stream;
   - exact trigger length in RTL simulation;
   - retained MAC lanes and monotonic resource scaling;
   - routed timing at the actual target clock;
   - no unexpected LED/USB/PLL switching during measurement.
6. Implement literal and adapted attack modes plus unit tests.
7. Generate all report numbers and figures from manifests rather than hard-coded text.

## Hardware reconnection phase

### Stage 0 - document the bench

Record board revision, FPGA ID, installed shunt value, JP7 state, decoupling population,
VCCINT, power supply, X4 cable/termination, CW-Lite firmware/serial, ADC gain, PLL outputs,
clock slew, temperature, USB-clock state, and photographs.

Use X4 to CW-Lite MEASURE as the primary path. Disable unused PLL outputs and the USB
clock during the measurement interval. Keep LEDs static. Check clipping and quantization
at every gain; CW-Lite samples are ADC units, not watts.

### Stage 1 - identify the CW305/CW-Lite channel

Build a small calibration bitstream with a controllable toggle bank. Apply:

- all-off and all-on steps;
- isolated one-cycle and multi-cycle bursts;
- PRBS activity;
- frequency sweeps;
- all-zero, all-255, checkerboard, row/column stripe, impulse, ramp, and random images.

Estimate noise PSD, drift, phase stability, impulse response, and frequency response of
the complete CW305 PDN + X4 + CW-Lite path. Do not assume 250 Hz or the paper's RC tail.

Model the capture as:

```
y = H p + b + noise
```

where `p` is latent per-cycle activity and `H` is the measured convolution matrix.
Recover `p` with a profiling-only tuned Wiener/Tikhonov inverse or use the calibrated
impulse response as a matched filter. Compare signed sum, RMS, peak-to-peak, matched
amplitude, and PCA waveform scores.

### Stage 2 - acquisition comparison

Primary mode:

- synchronous x4;
- target near 20-25 MHz so ADC is near 80-100 MS/s;
- sweep CW-Lite ADC phase;
- keep all four samples per clock initially.

Equivalent-time option when replay is permitted:

- repeat the deterministic operation at multiple calibrated ADC phases;
- interleave phase-shifted synchronous samples;
- report this as replay-assisted, not single-shot.

Secondary async mode:

- target 5 or 10 MHz and ADC near 105 MS/s;
- save every raw repeat;
- estimate global phase and period for each repeat;
- resample before combining.

Include presamples and a quiet settling interval after initialization. Capture the full
physical span and store the valid-cycle map.

### Stage 3 - leakage qualification pilot

Before another 500-image run, use 20-50 controlled inputs and hundreds of unaveraged
repeats for paired F1/F16 captures. Randomize input, kernel, and fanout order and insert
periodic reference traces.

Required diagnostics:

- fixed-versus-random TVLA with confirmation on an independent subset;
- per-cycle SNR/NICV and repeatability;
- cross-validated prediction of the exact synthesized transition model;
- same-image/different-kernel discrimination;
- single-pixel perturbation localization;
- shuffled label, kernel, cycle, and trace negative controls;
- fanout scaling versus post-route resource and switching evidence.

Proceed only if held-out metrics exceed shuffled controls with confidence intervals and
the leakage localizes to the intended cycles. Do not use a correlation with raw window
magnitude as the decisive test; the relevant model is state transition/Hamming activity.

### Stage 4 - preregistered 300/200 evaluation

- Select a fixed, stratified, seeded 500-image subset.
- Split the 300 profiling images internally into fit/validation for all preprocessing,
  delta, filters, and model choices.
- Rebuild the final template with all 300 only after choices are frozen.
- Keep the 200 evaluation images untouched.
- Primary online result: one trace per inference.
- Diagnostic averaging curve: N=1,2,5,10,20,50,100.
- Repeat across sessions/days and, if possible, place/route seeds.

Report:

- paper metrics with image-level bootstrap confidence intervals;
- all-black, all-background, majority-class, and shuffled controls;
- candidate count, strict-intersection count, no-match rate, and genuine-candidate
  recall (paper Table 2 equivalents);
- per-class confusion and conditional recognition where the golden classifier gets the
  original image right;
- exact acquisition count and whether the unknown input was replayed.

## Immutable run format

Every run should contain:

- run UUID and schema version;
- git commit, dirty diff hash, and hashes of capture/analysis scripts;
- HLS/Vivado versions, parameters, build seed, source/RTL/bitstream SHA-256;
- kernel/image dataset hashes and exact split indices;
- board/scope serials, firmware, actual clocks, lock state, gain, trigger definition,
  supply and temperature;
- randomized acquisition order and repetition index;
- raw samples, not only averages;
- start/end timestamps, expected/completed/failed capture counts, and per-file checksums;
- a completion marker written only after validation.

Resume only when the immutable manifest matches exactly. Cache keys must include trace
content, metadata, code, and bitstream hashes.

## Immediate next actions

1. Do not run another large capture yet.
2. Treat `traces_physical_v1` and its results as invalid.
3. Downgrade `FINAL_RESULTS.md` from a proven hardware-floor conclusion to an
   inconclusive acquisition result.
4. Complete the offline fixes and clean rebuilds.
5. When the boards return, run the calibration/pilot gates before the final 300/200
   experiment.

