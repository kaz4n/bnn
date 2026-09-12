# RGB input recovery: evidence assessment, research gap, and project claim audit

Prepared 11 September 2026. Read-only review. No bitstream was programmed, no FPGA
setting was altered, no software was installed and no experiment was started.

## Label legend

Every substantive statement carries one of these tags.

| Tag | Meaning |
|---|---|
| `[PE]` | Published evidence — stated or measured in a cited source I read in full |
| `[AB]` | Abstract only — I read the abstract, not the body; treat as unverified |
| `[PA]` | Project artifact — read from a file in this repository |
| `[INF]` | My inference from `[PE]`/`[PA]`, not itself measured |
| `[UP]` | Untested proposal — would have to be measured to be a claim |
| `[NR]` | Not read — cited in a source I read, but I did not access the original |

Sources read in full: the two supplied PDFs, plus downloaded copies of *Leaky Nets* and
Sanjaya et al.'s survey. Sources read as abstract only, or not at all, are marked.

### Requested deliverable → section map

| You asked for | Section here |
|---|---|
| A. Concise feasibility assessment | **§A** |
| B. Comparison table of the two papers | **§B** |
| C. Source-backed account of the RGB research gap | **§C** (literature backing it in **§D**) |
| D. Defensive research proposal with explicit limitations | **§E** (limitations in §E.7) |
| E. Unresolved questions affecting the conclusions | **§G** |
| Task 5 — audit of existing project claims | **§F** |

---

# A. Feasibility assessment

**The short answer: extending these two papers from grayscale to RGB is a genuine,
unclaimed research gap, and it is the right thing to investigate. But neither paper
provides evidence that it will succeed, and your specific bench has a documented
constraint that is independent of the color question and compounds with it.**

Five findings drive this.

**1. Neither paper ever exercised a multi-channel first layer.** `[PE]` Both
deliberately set input channels to one. Power2Picture §III-B: the authors' "only
modification to the original model is a reduction of the input channels to a single
channel." I Know What You See §6.3 sets "input channel to 1." So the leakage models in
both papers were never tested against the operation that RGB introduces.

**2. The older paper names color as explicitly untested future work.** `[PE]` I Know
What You See, Appendix B, states its images are "either quasi-binary (MNIST) or simple
gray-scale images," and that targeting "images with multiple input channel (e.g., color
images)" would require more enrolled data. CIFAR-10 and ImageNet validation is listed as
planned, not performed. The authors do not claim color recovery.

**3. Color input recovery from a physical side channel *has* been demonstrated — but by
a different mechanism, on different hardware.** `[PE]` Maji, Banerjee and Chandrakasan
(*Leaky Nets*) recover CIFAR-10 and ImageNet inputs with "Exact input recovery [...] in
all cases" (Fig. 13). This is a real result on color images and must not be waved away.
It is also not this problem: the platform is a microcontroller (ATmega328P, ARM
Cortex-M0+, RISC-V RV32IM), and the channel is principally *timing* — of floating-point
multiplication and integer-to-float conversion — plus SPA on the conditional subtractions
inside a fixed-point normalization divide (§VII-A and §VII-B respectively). `[INF]` That
channel exposes each input operand through its own instruction — §VII-A recovers the
input mantissa by matching a per-operand timing LUT, and §VII-B reads the quotient bits
of a per-input normalization divide — so per-channel values never merge; it is closer to
reading a value than to estimating it from aggregate switching power. The FPGA
accelerator case is the opposite: one accumulation absorbs all channels. So *Leaky Nets*
shows color recovery is not impossible in general, while leaving the FPGA
power-amplitude question open.

**4. The structural obstacle is channel summing, and it is not fixed by a bigger model.**
`[INF]` A first-layer convolution accumulates across input channels. For the attacked
geometry, one accumulation collapses 9 values (1×3×3) into a scalar today, and would
collapse 27 (3×3×3) under RGB — while the observable dimensionality (kernels × cycles)
does not grow. Unknowns scale as arithmetic, not conjecture: 784 at 28×28 grayscale,
3,072 at 32×32 RGB, 12,288 at 64×64, 49,152 at 128×128 — the largest size you named
carries 62.7× the unknowns of the published setting, against an observation budget that
is unchanged. Whether the residual color information is sufficient is an empirical
question. Nothing in either paper answers it, and I will not predict a number.

**5. Your bench has a separate, documented constraint.** `[PA]` `FINAL_RESULTS.md`
records that the CW-Lite samples at 105 MS/s, 4 samples per cycle, against the 2.5 GS/s
oscilloscope used in I Know What You See, whose §5 recovers per-cycle power by
curve-fitting the RC transient of each cycle. `[PE]` That per-cycle extraction is the
front end the template attack consumes. `[INF]` This constraint is orthogonal to channel
separability — it limits how cleanly any per-cycle quantity can be extracted, before
color is considered — and the two compound. `[PA]` Your project worked around it rather
than solving it, and the workaround matters to any RGB claim (see audit Finding 3).

**Assessment.** `[INF]` The gap is real and worth stating as a research question. The
honest framing is a *falsification* study — does channel-summed first-layer power retain
per-channel information at all, and how would you tell — not an assumption that scaling
up the reconstruction network closes it. `[UP]` Any such study on this bench should
expect the sampling constraint to bound results independently of color, and should be
designed so the two limits can be told apart.

---

# B. Comparison of the two supplied papers

| | **I Know What You See** (Wei, Luo, Li, Liu, Xu) | **Power2Picture** (Huegle, Gotthard, Meyers, Krautter, Gnad, Tahoori) |
|---|---|---|
| Venue / year `[PE]` | ACSAC '18; arXiv:1803.05847v2 (29 Nov 2019) | FCCM 2023, pp. 155–161, DOI 10.1109/FCCM57271.2023.00025 |
| Datasets `[PE]` | MNIST (§6.3); one DDSM mammography image, resized 168×84 (App. B, Fig. 10) | MNIST and Fashion-MNIST (§IV), 10k test / 60k train |
| Dimensions `[PE]` | 28×28 | 28×28 |
| Color representation `[PE]` | Single channel; "input channel to 1" (§6.3). App. B: images are "quasi-binary (MNIST) or simple gray-scale" | Single channel; input channels reduced to one because "both of our evaluated datasets contain only grayscale images" (§III-B). Generator output layer emits 1 channel (Table I) |
| Measurement location `[PE]` | **External.** Probe on the FPGA power supply line, through the board's amplifier (§5.1) | **On-chip.** TDC voltage sensors co-located with the victim on the same fabric (§II-A, Fig. 2); five sensors averaged |
| Hardware `[PE]` | Xilinx Spartan-6 LX75 on SAKURA-G; Tektronix MDO3034 at 2.5 GS/s (§4) | PYNQ-Z1 (Z-7020) main platform; ZCU104 for portability. Accelerator 50 MHz, TDCs 100 MHz (§IV) |
| Victim design `[PE]` | Binarized CNN on the Zhao et al. line-buffer accelerator; 64 first-layer kernels; 3×3 and 5×5 models (Table 1) | Quantized LeNet from Brevitas, deployed via FINN; 4-bit activations, 8-bit bias (§III-B) |
| Assumed knowledge `[PE]` | First-layer filter size, number of input and output feature maps, image size. **Not** weights or model parameters (§3) | Architecture known; a profiling instance available. Attack still works when victim weights are retrained differently (Table II, row 4) |
| Profiling required? `[PE]` | **Split.** Passive adversary (§6, background detection by thresholding) is non-profiled. Active adversary (§7, power template) is profiled: 300 template / 200 evaluation images | **Yes, always.** The generator is trained by supervised regression on known input/trace pairs (§III) |
| Metrics `[PE]` | Pixel-level accuracy as background-marker agreement (Eq. 5); pixel-level distance as L2 over 0–255 (Eq. 7); recognition accuracy against a 99.2% MLP reference | Recognition accuracy (reclassification by the original net); MSSIM, 11-px window; pixel-level distance (Eq. 2) |
| Headline results `[PE]` | §6 background: pixel acc 86.2% (3×3) / 74.6% (5×5); recog 81.6% / 64.6%. §7 template: recog **89.8%** (3×3) / 79.0% (5×5); pixel distance 1.65 | MNIST PYNQ-Z1 25 °C: recog **96.4%** quant / 95.0% non-quant, MSSIM 0.62, pixel distance 32/28. Fashion-MNIST: 53.0/65.0%, MSSIM 0.38. ZCU104: 55.1/49.8%, MSSIM 0.49 |
| Generalization *demonstrated* `[PE]` | Across kernel size; across two models; one cross-dataset qualitative example (mammography) with the template reused unchanged | Across two boards, two datasets, two FPGA families; across recompiled place-and-route; across 0/25/70 °C; across retrained victim weights (92.8% recog) |
| Generalization *not* tested | `[PE]` Color/multi-channel, CIFAR-10, ImageNet — named as future work. Non-line-buffer designs, GPU/TPU. Profiling from a different distribution. `[INF]` Larger images; the mammography case is a single qualitative figure with no metric | `[PE]` §VI concedes the generator parameter space "needs to be explored and optimized in future works, to recover inputs from even more complex architectures and datasets." `[INF]` No RGB, no resolution above 28×28, no non-FINN/LeNet victim, no quantitative cross-dataset transfer |
| Stated countermeasures `[PE]` | Random masking; random scheduling of kernel order (App. B) | None proposed; §VII defers defense to future work |

### What the metrics actually establish

`[PE]` **Both papers independently show recognition accuracy and pixel fidelity are
decoupled, measuring it in opposite directions.**

- I Know What You See, Fig. 8: with the reconstruction algorithm, pixel-level distance is
  1.65 and recognition is 89.8%. Without it, pixel distance is 2.98 and recognition
  collapses to **15%**. The authors describe *both* pixel distances as "quite close to
  the genuine image considering the pixel value range is 0 to 255." A near-equal mean
  pixel error therefore accompanies a 75-point recognition swing.
- Power2Picture §V-A, from the other side: "our mean pixel-level distance is worse than
  the results presented in [12], but our recognition accuracy is significantly higher."

`[INF]` These are the authors' own measurements, and together they establish that neither
metric substitutes for the other. Recognition accuracy measures whether a classifier
assigns the same label — it is dominated by a few structurally critical pixels, which is
exactly why Power2Picture's Fig. 5 and Fig. 8 show recovered digits and pullovers that
are visually close to the original yet misclassified. Recognition accuracy is not a
percentage of correctly recovered pixel values, and MSSIM is a structural-similarity
score, not a color-fidelity score.

`[INF]` A further limit on recognition accuracy as evidence: it is computed by feeding
recovered images to a classifier trained on the *same dataset*. A reconstruction that
regresses toward a class prototype — which a supervised generator trained on that dataset
is pressured to do — can score well on recognition while being wrong about the specific
input. Neither paper reports a control separating these.

---

# C. The RGB research gap

### C.1 Why recognizable grayscale recovery does not establish RGB fidelity

`[PE]` The published claim is recovery of a single-channel, 28×28, quasi-binary or
smooth-grayscale image, evaluated mainly by whether a same-dataset classifier still
assigns the right label. `[INF]` Four distinct things would have to hold for that to
extend to RGB, and none is tested by either paper:

1. **Per-channel separability.** `[INF]` The first-layer accumulation sums over input
   channels. A single accumulated value is consistent with many (R,G,B) triples — the
   channels enter through a weighted sum, so recovering *a* consistent triple is not
   recovering *the* triple.

   > **Correction, 11 September 2026 (same day).** This point originally continued "…so
   > channel identity is not separately observable from one accumulation." That is too
   > strong and is now measured to be wrong in general. The Phase 0 probe in
   > `experiments/rgb_sca/` shows a channel permutation *does* change the accumulator
   > whenever the kernel's per-channel weight planes differ, which trained first-layer
   > kernels generically do; identity becomes unobservable only for channel-*symmetric*
   > kernels, and under a channel-serial dataflow it survives regardless of weights. The
   > obstacle is real but **kernel- and dataflow-dependent, not universal**. §C.5's
   > conclusions are unaffected — the supplied papers still establish nothing about RGB —
   > but the mechanism stated here was wrong. See `experiments/rgb_sca/README.md`.
2. **Observation budget.** `[INF]` The template in I Know What You See maps a 3×3 patch
   to a 9-kernel power feature vector (§7.3). Under RGB the patch carries 27 values while
   the feature-vector length is set by the number of probed kernels and cycles, which
   does not grow because the input gained channels.
3. **Intensity range.** `[PE]` MNIST is described by the authors themselves as
   quasi-binary, and the §6 passive attack is a *background/foreground threshold* — a
   binary decision per pixel. `[INF]` Nothing in a binary foreground decision extends to
   estimating three 8-bit values.
4. **Scale.** `[INF]` 784 unknowns become 3,072 / 12,288 / 49,152 at 32/64/128 px RGB.

### C.2 Information loss, quantization, and ambiguity

`[PE]` Power2Picture §III-B states the constraint plainly: the victim "operates on
quantized input images," so "the accelerator and the measured power traces can only
contain information about the quantized inputs" — 4-bit activations in their setup.
`[INF]` The measurable upper bound on recoverable input precision is set by the
accelerator's own input quantization, before any measurement noise. For RGB this binds
three times over, and a reconstruction that outputs smooth 8-bit-per-channel color is
presenting precision the channel cannot carry.

`[INF]` Ambiguity compounds: with channel summing, the map from input to observation is
many-to-one by construction. Reconstruction under a many-to-one forward map is determined
only up to the null space, and what fills that null space is the *prior* — which in a
learned reconstructor is the training distribution.

### C.3 Why a plausible generated image may differ from the actual input

`[INF]` This is the central epistemic risk and it gets worse in color. A supervised
generative reconstructor trained on a dataset learns that dataset's statistics. Where the
measurement is uninformative, the model's best expected-error strategy is to emit the
dataset's typical content. The output is then plausible, well-scoring on structural
metrics, and *not evidence about the input*.

`[PE]` The published results are consistent with this being a live effect rather than a
hypothetical one. Power2Picture's Fashion-MNIST recognition (53.0–65.0%, MSSIM 0.38) is
far below its MNIST result (96.4%, MSSIM 0.62) on the same platform and pipeline; the
authors attribute this to "the greater importance of more fine grained pixel values for
the Fashion-MNIST dataset." `[INF]` That is the expected direction if what is recovered
degrades as the input departs from a sparse, near-binary, highly stereotyped
distribution. RGB photographs depart from it much further than Fashion-MNIST does.

`[INF]` For color specifically, plausibility is especially cheap: natural-image color is
strongly predictable from luminance and semantics (sky, skin, foliage). A model that
recovers structure from the side channel and *colorizes it from the prior* would produce
convincing RGB output while carrying little or no measured color information. Any RGB
claim has to rule this out; none of the supplied evidence does.

### C.4 Why three channels or higher resolution is not evidence

`[INF]` Producing a three-channel file is a property of the output layer, not of the
measurement — the generator in Power2Picture emits one channel because its last
transposed convolution has one output feature map (Table I); setting that to three
changes the tensor shape and nothing about what the trace contains. Likewise, output
resolution is set by the transposed-convolution stride, so upsampling produces more
pixels without more measured information. `[INF]` Neither a three-channel output nor a
larger output demonstrates that per-channel or fine-grained information was recovered;
both are decisions made on the reconstruction side of the pipeline.

### C.5 Conclusions that remain unsupported by the supplied papers

`[INF]` The supplied papers do **not** support any of these:

- That RGB or any multi-channel input has been recovered from FPGA accelerator power.
- That color information survives a channel-summed first-layer accumulation.
- That results at 28×28 predict results at 32×32, 64×64 or 128×128.
- That recognition accuracy implies pixel or color fidelity — both papers measure the
  opposite (§B).
- That a larger or more modern reconstruction network closes the gap. `[PE]`
  Power2Picture §VI calls this out as unexplored, not as solved.
- That an attack transfers to inputs drawn from a different distribution than the
  profiling set. `[PE]` I Know What You See App. B names this as a limitation; the
  mammography example reuses the MNIST template but is reported qualitatively, with no
  metric.

---

# D. Relevant primary literature

Sorted by the categories you asked for. `[NR]` marks works I did not access directly.

### D.1 Physical hardware demonstrations

| Work | Platform / channel | Inputs | Status |
|---|---|---|---|
| Wei et al., ACSAC '18 (supplied) `[PE]` | Spartan-6 / SAKURA-G, external scope 2.5 GS/s | MNIST 28×28 grayscale; 1 mammography image | Read in full |
| Huegle et al., FCCM 2023 (supplied) `[PE]` | PYNQ-Z1, ZCU104; on-chip TDC | MNIST, Fashion-MNIST, grayscale | Read in full |
| Moini et al., *Power Side-Channel Attacks on BNN Accelerators in Remote FPGAs*, IEEE JETCAS 11(2):357–370, 2021; arXiv:2011.07603 `[AB]` | Xilinx FPGAs **and AWS F1 cloud instances**; TDC voltage sensors | MNIST grayscale only; no color discussion in the abstract | **Abstract only.** arXiv abstract reports max normalized cross-correlation 79% local / 72% on F1; a search snippet of another version reports 84%/77%. I could not reconcile the versions and did not read the body |
| Maji, Banerjee & Chandrakasan, *Leaky Nets*, IEEE IoT-J 2021; arXiv:2103.14739 `[PE]` | **Microcontrollers** (ATmega328P, Cortex-M0+, RISC-V RV32IM); **timing** side-channel + SPA | **MNIST, CIFAR-10 and ImageNet — color included.** "Exact input recovery was achieved in all cases" (Fig. 13) | Read in full. **The one color result; its mechanism differs — see §A.3** |
| Schellenberg, Gnad, Moradi & Tahoori, *An inside job: Remote power analysis attacks on FPGAs*, DATE 2018 `[NR]` | On-chip TDC sensor design | Not an input-recovery paper | Cited as the sensor Power2Picture builds on (§II-A) |

`[INF]` Moini et al. is the most important work for your framing after the two supplied
papers: it is the only prior *remote* input-recovery result, it is what Power2Picture
critiques in §II-C (for requiring over 1000× averaging and for not reporting recognition
or pixel-accuracy metrics), and it is grayscale MNIST. I recommend reading the body
before relying on its numbers — I did not.

### D.2 Simulation-only or unconfirmed-modality results

- Wang, Wu, Park, Yoo, Wang, Eshraghian & Lu, *PowerGAN: A Machine Learning Approach for
  Power Side-Channel Attack on Compute-in-Memory Accelerators*, Advanced Intelligent
  Systems 5(12):2300313, 2023; arXiv:2304.11056. `[AB]` GAN-based input reconstruction
  against an analog compute-in-memory U-Net inference chip, demonstrated on **MRI medical
  images** — grayscale. `[AB]` Reported robust at 20% noise standard deviation.
  **Abstract only; I could not determine from the abstract whether the power data is
  physically measured or simulated, and I do not claim either.** Worth resolving, because
  a simulated-power result carries different weight than a measured one.

`[INF]` I found no simulation-only FPGA *color* input-recovery result either. The absence
is itself informative: the gap is not that color has been simulated but not measured — it
appears not to have been reported at all for this channel.

### D.3 Defensive evaluations and countermeasures

- Dubey, Cammarota & Aysu, *BoMaNet: Boolean masking of an entire neural network*,
  ICCAD 2020. `[NR]` Cited in Power2Picture [20].
- Glamočanin, Coulon, Regazzoni & Stojilović, *Are cloud FPGAs really vulnerable to power
  analysis attacks?*, DATE 2020. `[NR]` Cited in Power2Picture [19]. `[INF]` A
  skeptical-evaluation paper, and therefore directly relevant as a model for how to frame
  a falsification study.
- *SHIELD: An Adaptive and Lightweight Defense against the Remote Power Side-Channel
  Attacks on Multi-tenant FPGAs*, arXiv:2303.06486. `[NR]` Surfaced in search; not read.
- `[PE]` Countermeasures proposed inside I Know What You See App. B: random masking of
  pixel values before convolution, and random scheduling of kernel execution order — the
  latter aimed specifically at defeating the active adversary's multi-kernel feature
  vector.
- `[PE]` Power2Picture proposes no countermeasure; §VII defers it.

### D.4 Intentional computational-imaging measurements — controls, not evidence

`[PA]` These are already recorded in `report/RGB_PAPER_SCOPE_AND_METHOD_OPTIONS_2026-09-11.md`
and I am preserving their labeling. **They are a different measurement problem and none of
their results are evidence about power side channels.**

- Hoshi et al., *Real-time single-pixel imaging using a system on a chip FPGA*, Scientific
  Reports, 2022. `[NR]` FPGA acquisition and reconstruction at 128×128 via an external
  photodetector and ADC — a deliberate optical measurement.
- Welsh et al., *Fast full-color computational imaging with single-pixel detectors*, 2013.
  `[NR]` Full-color reconstruction from **three spectrally filtered measurement
  channels** — i.e. color is recovered because color was separately measured.

`[INF]` Welsh et al. is a useful contrast for exactly the reason it must not be cited as
support: it obtains color by giving color its own measurement channels. The power
side-channel setting offers no equivalent — the channels are summed, not separated.

### D.5 Hypotheses and stated future work

- `[PE]` I Know What You See, App. B: multi-channel/color inputs and cross-distribution
  profiling named as open; PCA for feature compression and SVM/random-forest candidate
  selection proposed; CIFAR-10 and ImageNet named as future validation targets.
- `[PE]` Power2Picture, §VI: generator parameter space "needs to be explored and optimized
  in future works, to recover inputs from even more complex architectures and datasets";
  §VII defers countermeasure research.
- `[PE]` Sanjaya, Jayasena & Mishra, *Application-Specific Power Side-Channel Attacks and
  Countermeasures: A Survey*, Proceedings of the IEEE (arXiv:2512.23785). Read in full.
  Its input-recovery coverage lists Wei (physical, FPGA, CNN, input data), Moini (remote,
  FPGA, BNN, input data), Huegle/Power2Picture (remote, FPGA, LeNet, input data, GAN) and
  Wang/PowerGAN (physical probing, ASIC, U-Net, input data, GAN) in its Table VII.
  `[INF]` **No entry in that table reports color or multi-channel input recovery from an
  accelerator power side channel.** I searched the survey text for color/RGB/multi-channel
  in connection with input recovery and found none.

---

# E. Defensive research proposal

`[UP]` Everything in this section is proposed, not performed. It is framed as a
privacy-evaluation and falsification study. It deliberately contains no attack
construction, parameter tuning, or optimization guidance.

### E.1 Research questions

- **RQ1.** Does the power signature of a channel-summed first-layer convolution on an FPGA
  accelerator retain *any* per-channel information about an RGB input, beyond what the
  dataset prior supplies?
- **RQ2.** If some information is retained, does it support *faithful* per-channel
  recovery, or only *recognizable* content?
- **RQ3.** How much of any apparent RGB recovery is attributable to the reconstruction
  model's prior rather than to the measurement?
- **RQ4.** Do the metrics used in the existing literature (recognition accuracy, MSSIM,
  mean pixel distance) remain adequate to answer RQ2 for color, or do they systematically
  overstate it?

`[INF]` RQ3 is the one the existing literature does not address and the one that
determines whether any positive RGB result would be believable.

### E.2 What would count as evidence for, and against

**For the privacy risk — all of these would need to hold:**

- Per-channel error substantially below what a prior-only baseline achieves, where the
  prior-only baseline is a reconstructor given *no trace* (or a shuffled trace) and the
  same training data. `[INF]` This baseline is the single most important control and it is
  absent from both supplied papers.
- The advantage persists on inputs drawn from outside the profiling distribution.
- The advantage persists under a channel-permutation control: if swapping the R and B
  channels of the true input does not change the reconstruction's error, the pipeline is
  not recovering channel identity.
- Results reproduce across a re-programmed design and a separate device, as Power2Picture
  demonstrated for grayscale `[PE]`.

**Against the privacy risk — any of these:**

- Per-channel error statistically indistinguishable from the prior-only baseline.
- Recovered color that tracks the dataset's class-conditional mean color rather than the
  specific input's color.
- Advantage that vanishes once the profiling and evaluation distributions differ.
- Advantage that vanishes at the input quantization the accelerator actually uses.

### E.3 Distinguishing memorization, recognizable content, and faithful reconstruction

`[UP]` Report these as three separate results, never collapsed into one headline:

1. **Memorization / prior** — measured by the prior-only baseline above, plus a
   nearest-neighbor check: how often is the reconstruction closer to some *training* image
   than to the true input? `[INF]` A pipeline that frequently reproduces a training
   neighbor is exhibiting memorization, and both supplied papers' recognition metric would
   score that as success.
2. **Recognizable content** — classifier agreement, i.e. the existing recognition accuracy.
   `[PE]` Report it with the original-data accuracy alongside, as Power2Picture does in
   Table II, since it is bounded by the classifier.
3. **Faithful reconstruction** — per-channel MAE/PSNR reported *per channel and not
   pooled*, plus a color-difference measure, plus the channel-permutation control.
   `[INF]` Pooling RGB into one PSNR hides the failure mode of interest; a reconstruction
   that gets luminance right and chrominance wrong scores well pooled.

### E.4 Reporting uncertainty and dataset limitations

`[UP]`

- State the number of *independent* scenes, not the number of evaluations. `[PA]` Your own
  `EVALUATION_PROTOCOL.md` already applies this discipline — "The three sizes reuse the
  same source scenes; they are not independent statistical samples" — and it should carry
  over.
- Report confidence intervals over scenes; at the scene counts practical here, differences
  of a few points are not resolvable. `[PA]` The existing `trained-kernels-beat-onehot`
  note already applies this reasoning at n=30.
- Report failed and degenerate cases rather than omitting them. `[PA]` Precedent exists in
  this repo: `AMP_SWEEP_VOID.md` and the "void: no gate" row retained in `tab_p2p.tex`.
- Distinguish pooled from per-image aggregation explicitly.
- State the accelerator's input quantization, since `[PE]` it bounds recoverable precision
  (Power2Picture §III-B).

### E.5 How countermeasure claims should be assessed

`[INF]` A countermeasure evaluated only against the specific reconstruction pipeline used
in the same paper establishes little — the attack side is a free parameter. Claims should
state: what adversary capability is assumed; whether the evaluation re-profiles *against*
the defended design (a defense that only breaks a template built on the undefended design
is not evaluated); the cost in area, power and accuracy; and whether the reported
reduction is in recognition, in pixel fidelity, or in both, since §B shows those move
independently. `[PE]` For the specific countermeasures named in I Know What You See
App. B, random scheduling targets the active adversary's cross-kernel feature vector
directly, so its evaluation must include an adversary that does not depend on kernel order.

### E.6 Results that would justify narrowing or rejecting the hypothesis

`[UP]` Pre-commit to these before measuring:

- No advantage over the prior-only baseline on per-channel error → **reject** RQ1 for this
  geometry and report it as a negative result. `[INF]` A credible negative here is
  publishable and is the most likely honest outcome to prepare for.
- Advantage present at 32×32 but absent at 64×64 and 128×128 → **narrow** the claim to a
  size regime and report the boundary.
- Advantage present only when profiling and evaluation share a distribution → **narrow** to
  a same-distribution threat model and say so.
- Advantage present in luminance but not chrominance → **narrow** to "grayscale recovery
  from an RGB-input accelerator," which is a legitimate and distinct finding, and
  explicitly *not* RGB reconstruction.

### E.7 Limitations of this proposal

`[INF]` Stated up front so they are not discovered later: the bench sampling constraint
(§A.5) may bound results independently of color, and a negative result could be caused by
it rather than by channel summing — the design must be able to separate these, and I do
not currently see a clean way to do so on this hardware alone. The scene counts practical
on this bench are small. A profiling-capable adversary with a comparable accelerator is
assumed, per your stated research assumptions, which makes this an upper-bound privacy
assessment rather than a realistic-adversary one.

---

# F. Audit of existing project claims

`[PA]` Read-only inspection. Findings ordered by how much they affect conclusions.

### Finding 1 — `FINAL_RESULTS.md` contradicts the 30 August 2026 results (hardware vs. simulation)

**Severity: high. This is the confusion category you asked about, and it is live.**

`[PA]` `FINAL_RESULTS.md` states under "What did NOT work — real-silicon recovery" that
running the §6/§7 attacks on measured CW305 power "gave recog ~0.15 (paper ~0.90)," with
correlation stuck at 0.1–0.2, and credits only simulation with reproducing the paper
("P3 in simulation — **Reproduces the paper**").

`[PA]` `report/catchup_v2_2026-08-30/tables/tab_headline.tex` reports *measured hardware*
template results of recognition 0.90–1.00 (pixel accuracy 0.953–0.968, MSSIM 0.713–0.816,
n=40), and `tab_grey.tex` reports the grey design at MAE 9.55, correlation 0.945, MSSIM
0.842, recognition 0.98 (n=300). `[PA]` `tab_provenance.tex` records every one of those
captures as "not stock AES: yes" with 676/676 functional checks and 0 mismatches.

`[INF]` A reader opening `FINAL_RESULTS.md` — whose name and root-directory placement both
signal authority — gets the opposite conclusion from a reader opening the August catch-up
report. `FINAL_RESULTS.md` appears to predate the fix recorded in
`GAP_HARDWARE_START_2026-08-29.md` (the CW305 answering as the stock AES bitstream) and was
not retracted or annotated afterwards. `[PA]` `RO_CHANNEL_RESULTS.md` (14 July 2026) has
the same *dating* problem — it is undated relative to the August work and its channel
ranking was superseded by it — though its error runs in a different direction; see
Finding 6.

**Recommended (not applied):** add a dated superseded-by banner at the top of
`FINAL_RESULTS.md` and `RO_CHANNEL_RESULTS.md` pointing to the August results. I did not
edit these files.

### Finding 2 — the amplifier sweep invites the wrong inference in isolation

**Severity: medium-high, and it directly affects any feasibility argument.**

`[PA]` `tab_amp_sweep.tex` shows recognition falling 0.857 → 0.753 → 0.443 → 0.400 as
amplifier lanes fall 63 → 15 → 3 → 1 (4536 → 72 flops). Read alone, that says the attack
depends on the deliberate leakage amplifier.

`[PA]` It does not. `attack/hardtests_20260830/p2p_paper60k_noamp/summary.json`
(8 September 2026) records a **zero-amplifier** grey bitstream
(`cw305_leakage_greynoamp_d8.bit`, sha256 `c52613305b…`), real capture
(`capture_mode: live_cw305_chipwhisperer`, `input_mode: grey`, `n_images: 60000`), reaching
MSSIM 0.724 and recognition 0.746 on 2,000 held-out Fashion test images. The `_d2` variant
reaches MSSIM 0.762, recognition 0.807.

`[INF]` The sweep rows used small profiling sets while the zero-amplifier point used the
paper-scale 60k set, so amplifier width and profiling-set size are confounded in that
table. The defensible statement is that **the amplifier substitutes for profiling data,
not for feasibility** — and the two numbers should always be quoted together. `[PA]` This
is consistent with the project's own `leak-sink-must-be-registered` note.

### Finding 3 — victim fidelity: three separate facts, easily collapsed into one

`[INF]` Stated precisely, because overstating this is as wrong as ignoring it:

1. `[PA]` The convolution core is a port of the line-buffer accelerator that I Know What
   You See itself attacks (Zhao et al. / bnn-fpga). **Faithful.**
2. `[PA]` The CW305 wrapper adds a `leak_sink` output and optional amplifier lanes.
   **Not present in either paper's victim.** `[INF]` Any result quoted from an amplified
   build is from a design more leaky than a stock accelerator, and should say so.
3. `[PA]` A zero-amplifier build was measured (Finding 2), so the wrapper additions are not
   load-bearing for feasibility.

`[INF]` The accurate sentence is: "results on an amplified build overstate a stock
accelerator's leakage; a zero-amplifier build was separately measured and still recovers."
Not: "the victim is instrumented, so the result does not count."

### Finding 4 — no RGB power side-channel result exists in this project

`[PA]` Every power side-channel capture manifest I inspected records `input_mode` as
`binary` or `grey` — single channel. The grey design is the project's only non-binary input
mode, and it is 0–255 single-channel (`tab_grey.tex`: "Grey design (real 0–255 pixels)").
`[INF]` So the project's power side-channel line is, at best, **grayscale-equivalent to
the published work**, and has produced no color evidence. Any future write-up must not let
the presence of an RGB directory in the same repository imply otherwise.

### Finding 5 — intentional sampling vs. physical leakage is correctly separated (positive)

`[PA]` `experiments/rgb_instrumented/` is labeled correctly and repeatedly. Its `README.md`
opens by calling it "an intentional imaging experiment on CW305, separate from the existing
power side-channel work," states "No external ADC or physical leakage sensor is used," and
`EVALUATION_PROTOCOL.md` states "It does not measure physical power, electromagnetic or
timing leakage" and that a valid capture "demonstrates correct intentional
sampling/readback, not proof of any physical leakage-recovery claim."
`EXPERIMENT_RESULTS.md` repeats it: "This is an instrumented demosaicing benchmark, not
physical power side-channel reconstruction."

`[INF]` **No confusion found in this area.** The one residual risk is contextual rather
than textual: this is the only RGB work in the repository, it reports 27.86 dB PSNR at
128×128, and it sits beside a side-channel project whose goal is RGB recovery. A reader
skimming for "RGB results on CW305" could easily carry those numbers across. The current
labeling is the mitigation and should be preserved verbatim in any external write-up.
`[PA]` The existing `RGB_PAPER_SCOPE_AND_METHOD_OPTIONS_2026-09-11.md` already makes the
same point ("they do not establish power leakage or progress on the user's side-channel
objective").

### Finding 6 — on-FPGA sensing vs. external acquisition is tracked, but the two channels' history is easy to misread

`[PA]` The project has used both: an on-chip RO/TDC sensor read back over USB
(`RO_CHANNEL_RESULTS.md`, register 16) and external CW-Lite analog capture
(`trace_source: cw_lite_analog_sync_x4`). `[PA]` The July RO write-up concluded the on-chip
sensor was the better channel (~2× the external ADC), whereas the August headline results
come from the external channel. `[INF]` Both statements were true when written; together
they read as a contradiction unless dated. This matters for comparison with Power2Picture,
which is an **on-chip TDC** result — the project's best result is from the **external**
channel, so it is methodologically closer to I Know What You See than to Power2Picture,
regardless of which reconstruction algorithm is used.

### Finding 7 — recognition vs. pixel fidelity is reported correctly (positive)

`[PA]` `tab_grey.tex` and `tab_fashion_grey.tex` report MAE (0–255), correlation, MSSIM,
F1 and recognition side by side rather than leading with recognition alone.
`tab_headline.tex` additionally reports an "All-background" column (0.887–0.888) `[INF]` —
which functions as a trivial-baseline control for pixel accuracy, and is exactly the kind
of control §E.2 asks for. That practice is ahead of both supplied papers and should be kept
for any RGB work, extended to a per-channel and prior-only form.

### Finding 8 — voided results are retained rather than silently dropped (positive)

`[PA]` `AMP_SWEEP_VOID.md` documents that four supposedly different bitstreams were in fact
the same one (spread 0.4%, order backwards) because reprogramming silently failed, and
voids the derived numbers. `[PA]` `tab_p2p.tex` retains an "Earlier 9000-image run
*(void: no gate)*" row rather than deleting it. `[PA]` `EXPERIMENT_RESULTS.md` preserves
the failed first bulk-upload attempt. `[INF]` This is good practice and materially raises
my confidence in the surviving numbers.

### Finding 9 — attacker-knowledge assumptions, and what was actually measured

`[PA]` The active template route assumes a known kernel schedule and uses attacker-chosen
probe kernels (`ACTIVE_POWER_TEMPLATE_RESULTS.md`; `probe_kernels`, `kernel_index` in the
manifests). `[PE]` I Know What You See §3 does grant the active adversary arbitrary-input
profiling, so this is broadly faithful to the paper's threat model. `[PA]` The
deployed-kernel case *was* measured (27 August 2026, `--probe-kernels file`): trained
kernels scored pixel accuracy 96.8%, MSSIM 0.824 — better than one-hot probes at 94.7% /
0.725. `[PA]` The project's own note correctly cautions against ranking the two on
recognition at n=30. `[PA]` `tab_headline.tex` also reports the negative transfer case: a
one-hot template against trained traces collapses to recognition 0.20, MSSIM 0.094.
`[INF]` So the "one-hot crutch" caveat is resolved by measurement for this configuration —
state it as measured for this geometry, not as a general result.

---

# G. Unresolved questions that materially affect the conclusions

Ordered by how much the answer would change the assessment.

1. **Does any per-channel information survive the first-layer channel-summed
   accumulation?** `[INF]` This is the crux. If the answer is no, the hypothesis is dead at
   the architecture level regardless of bench, model or dataset, and no amount of
   reconstruction capacity changes it. Nothing in the supplied papers addresses it.
2. **Is the CW-Lite sampling constraint separable from the color question?** `[PA]`
   `FINAL_RESULTS.md` documents 4 samples/cycle vs. the paper's 2.5 GS/s with per-cycle RC
   curve-fitting. `[INF]` If a negative RGB result arrives, this bench cannot currently
   distinguish "color does not survive" from "this bench cannot extract it." A study design
   that cannot separate these produces an uninterpretable result.
3. **How much of any RGB recovery would be the dataset prior?** `[INF]` Unmeasured in every
   paper reviewed, including both supplied ones. Until a prior-only baseline is reported,
   no color result is interpretable.
4. **What does Moini et al. actually report?** `[PE]`/`[INF]` I read the arXiv abstract
   only, and found two different figure pairs (79%/72% vs. 84%/77%) across versions without
   resolving them. It is the closest prior art and the only other remote result; its body
   should be read before it is cited in any write-up.
5. **Is PowerGAN's power data measured or simulated?** `[PE]` The abstract does not say, and
   I did not read the body. `[INF]` It is the only other GAN-based input-recovery work on an
   accelerator, so its modality determines whether it belongs in §D.1 or §D.2.
6. **Does *Leaky Nets*' color result depend on per-operand timing granularity?** `[INF]` My
   reading of §VII says yes — mantissa recovery per operand and SPA on a per-input
   normalization divide — which is why it does not transfer to a channel-summed FPGA
   accumulation. This inference is load-bearing for §A.3 and deserves independent
   verification, since it is the single result that most complicates the gap claim.
7. **Does the input quantization of a realistic RGB accelerator leave enough precision to
   matter?** `[PE]` Power2Picture §III-B bounds recoverable precision by the accelerator's
   quantization (4-bit activations there). `[INF]` Unknown for an RGB design, and it caps
   any achievable color fidelity before measurement noise.
8. **Would a stock (non-amplified, no `leak_sink`) RGB design leak at all on this bench?**
   `[PA]` The zero-amplifier grayscale point exists (Finding 2), but `[INF]` it does not
   transfer automatically to a channel-summing design whose per-cycle activity differs.
9. **Is `FINAL_RESULTS.md` intended as superseded?** `[PA]` It is unannotated and sits in
   the repository root. `[INF]` Until answered, the project has two contradictory headline
   claims about whether hardware recovery works.

---

## Scope note

`[INF]` This document assesses evidence and proposes falsification criteria. It contains no
attack construction, no reconstruction pipeline, no parameter selection and no operational
procedure for recovering hidden inputs — including for the RGB case, which is the point at
which such material would be new rather than merely restated. Tuning parameters visible in
the repository were deliberately excluded from §E.
