# Fixed evaluation protocol

This experiment reconstructs RGB images from deliberately exported RGGB Bayer
samples. It does not measure physical power, electromagnetic or timing leakage.
The FPGA receives RGB8 pixels, selects R/G/G/B by coordinate parity, and stores
the resulting bytes in on-chip RAM. The host retrieves those bytes. Two of the
three color values at each pixel are missing from the reconstruction input.

Hardware acquisition is required for the headline experiment. Behavioral RTL
simulation and the explicit software model are development checks with separate
provenance; neither may be substituted when FPGA capture fails.

The fixed methods are nearest-neighbor interpolation, normalized bilinear
interpolation, graph-Laplacian inpainting with a ridge toward bilinear output,
and constrained isotropic TV regularization of bilinear output. All use exact
measured-sample constraints. Defaults are defined in reconstruction.py before
the physical evaluation; they are not tuned on the evaluation images. The latter
two are regularized interpolation methods, not universal inverse solvers for
unknown observations.

The four natural examples are astronaut, coffee, Chelsea and rocket from the
pinned scikit-image v0.18.3 example collection. Three diagnostic controls are
color bars, a smooth color gradient and a pixel-frequency pattern that exceeds
the Bayer color sampling bandwidth. Each is evaluated at 32, 64 and 128 pixels
square. A center crop and Lanczos resize define the natural RGB8 reference at
each size. All processing and errors use encoded RGB values, not linear light.
Natural and diagnostic results are reported separately. The three sizes reuse
the same source scenes; they are not independent statistical samples.

Captures contain only the mosaic and provenance. Reference PNGs are stored in a
separate truth directory and used for validation/scoring. Reconstruction APIs
receive the mosaic alone. Input and observation hashes, programmed bitstream
hash, dimensions, count, source, protocol status and hardware validation status
are recorded. A capture is valid only when every returned byte matches the
defined sampling operation, the frame is complete and the device reports no
protocol error. This demonstrates correct intentional sampling/readback, not
proof of any physical leakage-recovery claim.

Report RGB MSE and PSNR, per-channel MAE, missing-channel-only MSE/PSNR, exact
measurement consistency and wall-clock reconstruction time. Aggregate pooled
MSE PSNR is labeled separately from mean per-image PSNR. Perfect PSNR is stored
with an explicit infinity flag rather than invalid JSON. Measured values account
for one third of RGB entries, so missing-only scores matter. Record failed cases
and solver failures; never omit them from the run status. Export comparison
images after scoring the floating-point reconstruction; display PNGs are rounded
to RGB8.

The four natural images form an engineering pilot, not evidence of generalization
to arbitrary photography or proof that one method is universally best. No
training dataset or learned CNN is part of this first experiment. The Nyquist
control intentionally demonstrates lost information; it is not an acquisition
failure if the sampled bytes are correct and reconstruction is poor.

The requested skills have narrower scopes than image reconstruction:

- advanced-evaluation concerns LLM judging. Its evidence-before-claims and bias
  separation principles inform the report, but objective pixel errors determine
  this comparison; no LLM score is presented as image fidelity.
- reasoning-trace-optimizer informs the observable execute/check/fix loop. Only
  commands, failures and outcomes are recorded; no private reasoning traces or
  external model service are used.
- context-optimization informs compact result summaries and scoped delegation.
  No unmeasured token savings or cache changes are claimed.
- systemverilog supplies synchronous RTL and verification guidance.
