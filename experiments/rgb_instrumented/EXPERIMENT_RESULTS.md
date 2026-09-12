# First physical FPGA RGB reconstruction results

Completed 11 September 2026. The CW305 performed intentional RGGB sampling and
stored the measurements in FPGA RAM. The host reconstructed the missing color
values. This is an instrumented demosaicing benchmark, not physical power
side-channel reconstruction.

All 21 frames completed at 32×32, 64×64 and 128×128: four photographs and three
diagnostic patterns per size. All 150,528 returned samples matched the specified
sampling operation exactly. Every frame records source=fpga, complete hardware
status, zero mismatches, input/capture hashes and the programmed bitstream hash.
All 84 reconstruction runs succeeded and retained the measured values exactly.

The 128×128 natural-image comparison is:

| Method | Pooled RGB PSNR | Missing-channel PSNR | Mean host reconstruction time |
|---|---:|---:|---:|
| Bilinear |27.8609 dB|26.1000 dB|1.845 ms|
| Constrained TV |27.8217 dB|26.0608 dB|679.400 ms|
| Smooth ridge |27.2469 dB|25.4860 dB|18.006 ms|
| Nearest neighbor |23.4432 dB|21.6823 dB|2.129 ms|

Bilinear has the highest pooled score in this pilot and is a useful inexpensive
baseline. Its advantage over TV is only 0.0392 dB; TV scores higher on two of the
four scenes. These results do not establish a universal quality ranking.

At 32×32 and 64×64, bilinear pooled natural-image PSNR is 22.503 and 25.273 dB,
respectively. The larger grids preserve more detail in these resized scenes.
The four source photographs are reused at each size, so they do not constitute
twelve independent scenes. Times are single-call measurements on this host,
exclude acquisition, and should not be treated as a throughput benchmark.

The pixel-frequency diagnostic performs poorly for every method, despite
correct hardware samples. The sampling operation discards information needed to
recover its original colors. Its error is reported separately from the natural
images; averaging it with smooth gradients would hide the very different cases.

All scores use encoded RGB in [0,1], before display PNG rounding. One third of
values are directly observed, hence the separate missing-channel score. Methods
receive only the captured mosaic; reference images are used for validation and
scoring. Parameters were fixed before the physical evaluation, and no model was
trained on these images. CNNs, Deep Image Prior and diffusion have not been
implemented in this first experiment.

Verification completed: 66 Python tests; capture-core simulation at all three
sizes; final wrapper simulation with 18,169 minimum-profile USB reads and 17,408
verified bytes; zero implementation DRC violations. The final design uses 100
LUTs, 138 registers and four 36 Kb block RAMs. Setup slack is +0.541 ns and hold slack
is +0.128 ns under the model documented in hardware/TIMING.md. The external timing
allowances are assumptions, not full board-level characterization.

The first bulk-upload hardware attempt failed with zero recorded samples and is
preserved in results/physical_smoke_20260911/capture_manifest.json. A controlled
probe found short USB control transfers worked while the tested bulk transfers
did not. The final run uses 45-byte uploads and 32-byte readbacks. No generated
measurements were substituted. The explicitly labeled software development
check is in its own directory and contributes none of the headline scores.

Evidence is in results/physical_20260911: capture_manifest.json, captures/,
truth/, metrics.json, results.md, outputs/ and comparison.png. An independent
read-only review verified all 21 input/capture hashes, all 84 output consistency
checks, all 24 aggregation groups and the current bitstream hash. Build source
hashes, resource counts and limitations are in build/build_provenance.json.

The CW305 is currently programmed with this instrumented benchmark. Reproduction
commands are in README.md and run.ps1. The next useful algorithm comparison would
be a separately trained demosaicing model against the fixed bilinear baseline,
with more independent natural scenes; that work is not claimed here.

Handoff: objective was to implement and compare small RGB reconstruction methods
with physical FPGA acquisition. All changes for this turn are inside
experiments/rgb_instrumented/. The main evidence is the complete physical capture
and evaluation run above. Limitations are the intentional sampling model, small
scene set, single-call timing, and documented USB timing assumptions.
