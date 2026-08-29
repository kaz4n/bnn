# Active Power-Template Attack on CW305 + ChipWhisperer

This is the stricter active-template route for the MNIST side-channel demo.  It
is closer to the active attack in Wei et al., *I Know What You See*, than the
earlier direct one-hot equation decoder.

## Method

The attack has four separated stages:

1. **Profile stage**
   Known MNIST images are sent to the CW305 leakage bitstream and measured with
   ChipWhisperer.  The script builds a power template:

   ```text
   PT = {(3x3 patch bits, 9-kernel power feature vector)}
   ```

2. **Trace-only attack bundle**
   Attack traces are copied into a new directory with only signal arrays and
   capture metadata.  The original image and label are removed.

3. **Attack stage**
   The attacker reads only the trace-only bundle and the profile template.  For
   each convolution window, it searches the template by grouped power-vector
   distance, intersects candidate patch sets, and then uses greedy overlap
   consistency to choose one patch per window.

4. **Score stage**
   Ground truth is loaded only after reconstruction to compute pixel metrics and
   recognition accuracy.

This means the reconstruction stage does not read attacked images or labels.
The active/profiled knowledge is the known schedule, the chosen one-hot probe
kernels, and the profile template.

## Files

- Main script:
  `attack/active_power_template_cw305.py`
- Profile template:
  `attack/active_template_hw/profile_template.npz`
- Held-out trace-only bundle:
  `attack/active_template_hw/attack_trace_only/`
- Held-out truth file:
  `attack/active_template_hw/truth_eval.json`
- Held-out results:
  `attack/active_template_hw/results_delta1_g3_v2/`
- Fresh hardware trace-only bundle:
  `attack/active_template_hw/fresh2_trace_only/`
- Fresh hardware results:
  `attack/active_template_hw/fresh2_results_delta1_g3/`
- Figures:
  `report/paper_figures/fig_active_power_template_trace_only.png`
  `report/paper_figures/fig_active_template_fresh_hardware.png`

## Hardware evidence

The fresh validation capture was run on the board after the active-template
script was added.

```text
capture: images=3 kernels=9 avg=3 samples=22144 adc_locked=True
captured img0062 label=9
captured img0063 label=3
captured img0064 label=7
```

ChipWhisperer reported a firmware warning (`0.51.0`, latest `0.54.0`), but ADC
lock was true and the captures completed.

## Results

### Held-out real hardware traces

Profile: first 30 previously captured CW305/CW-Lite images.

Attack: next 30 images, copied into trace-only files before reconstruction.

```text
bit accuracy              95.62%
foreground precision      82.08%
foreground recall         78.62%
foreground F1             79.75%
foreground IoU            66.84%
patch exact               76.33%
recognition accuracy      96.67%  (29/30)
```

The original paper reports 89.8% recognition accuracy for the 3x3 active
power-template attack.  This run is above that recognition number, but it uses a
more controlled CW305 bitstream with one-hot probe kernels.

### Fresh hardware traces

Profile: same old profile template.

Attack: three new board captures, indices 62--64, copied into trace-only files.

```text
bit accuracy              94.22%
foreground precision      77.13%
foreground recall         74.28%
foreground F1             75.33%
foreground IoU            60.74%
patch exact               73.62%
recognition accuracy      100.00%  (3/3)
```

## Validation of trace-only property

Checked attack bundles:

```text
files_checked: 33
forbidden_key_hits: []
sample keys: repeats, traces, kernel_ids, samples_per_cycle, dwell,
             samples_per_window, presamples, trace_source
```

No attack `.npz` file contains `image`, `label`, `original`, `truth`, or
`true_codes`.

## Important limitation

This is not a passive unknown-model attack.  It is an active/profiled attack.
It uses controlled one-hot probe kernels and a leakage-enhanced CW305 bitstream,
because ChipWhisperer has lower analog bandwidth and memory than the
oscilloscope setup in the original paper.  The strong claim is:

> On this hardware setup, real FPGA power traces are enough to reconstruct
> recognizable MNIST inputs under an active power-template methodology.

