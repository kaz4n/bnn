# Instrumented RGB reconstruction experiment

This is an intentional imaging experiment on CW305, separate from the existing
power side-channel work. The FPGA selects one R/G/G/B sample per pixel and stores
those digital observations in on-chip RAM. Reconstruction runs on the host and
receives only the downloaded mosaic. The two missing channels at each pixel are
estimated. No external ADC or physical leakage sensor is used.

The first physical run is complete: 21 frames, 150,528 checked FPGA samples and
84 successful reconstructions. See EXPERIMENT_RESULTS.md and
results/physical_20260911/results.md for the measurements and comparison image.

Supported images are square RGB8 at 32, 64 and 128 pixels. A 128-pixel frame
streams 48 KiB of RGB input and produces 16 KiB of FPGA-buffered measurements.
The natural images are four pinned [scikit-image examples](https://scikit-image.org/docs/stable/api/skimage.data.html).
Numeric color bars, gradients and a deliberate aliasing pattern provide separate
diagnostic controls. Source URLs and SHA256 hashes are retained in captures.

The implemented reconstruction methods are nearest neighbor, normalized bilinear,
graph-Laplacian smoothing with ridge toward bilinear, and constrained TV. All
retain the measured samples exactly. The complete objectives and parameters are
in reconstruction.py. No CNN, diffusion model or hidden-input recovery is
implemented in this first experiment. See EVALUATION_PROTOCOL.md for fixed
metrics, independence limits and the requested skills' roles.

**Run on hardware**

From this repository in PowerShell:

```powershell
& './experiments/rgb_instrumented/run.ps1'
```

This builds and programs a new volatile CW305 design, replacing the FPGA's
currently running configuration. It creates a new physical run directory and
never substitutes simulated samples on failure. The existing application source
and bitstreams are not used by the experiment. The default Python and Vivado
paths match this machine; pass -Python and -Vivado to override them. Set the
CW305_USB_FRONTEND environment variable if the stock NewAE frontend source is
elsewhere. -SkipBuild reuses the experiment's existing bitstream.

The hardware target is xc7a100tftg256-2. The project uses the stock NewAE USB
register frontend, plus the independent rgb_bayer_capture core and wrapper.
The capture logic uses one USB clock domain. hardware/TIMING.md documents the
peripheral timing model and its limits. Vivado aborts bitstream generation if
the checked setup or hold path fails; normal bitstream DRC is retained.

The observed board/SDK combination accepted short USB control writes but did
not record the tested bulk uploads. The capture driver therefore uses 45-byte
control writes and 32-byte control reads. This is a measured compatibility
workaround; its exact firmware/SDK cause is not established. The board firmware
was not upgraded. Acquisition timing includes this transport overhead.

**Inspect results**

A completed run contains capture_manifest.json, captures/*.npz (mosaic plus
provenance), truth/*.png, outputs/<method>/*.png, metrics.json, results.md and
comparison.png. The capture manifest identifies the FPGA bitstream and records
sample counts, protocol status, observation hashes and byte-for-byte reference
validation. The metrics distinguish natural scenes from controls, include
missing-channel error and retain failed cases. Scores are computed before PNG
rounding. The small scene set is a pilot, not statistical evidence of broad
generalization. A poor aliasing-control reconstruction is expected when
sampling loses information.

To score an existing capture directory again:

```powershell
& 'C:/Users/narut/ChipWhisperer/cwenv/Scripts/python.exe' -m experiments.rgb_instrumented.evaluate --run-dir <existing-run-directory>
```

**Validate components**

```powershell
python -m pytest experiments/rgb_instrumented/tests -q
& './experiments/rgb_instrumented/sim/run_xsim.ps1'
& './experiments/rgb_instrumented/sim/run_wrapper_xsim.ps1'
```

The explicit --source software_model option in capture.py is for development
only and labels every record accordingly. Simulation runners and their detailed
protocol checks are documented in sim/README.md. Simulation is not a physical
capture result. Current Python tests use the system Python with pytest; physical
acquisition and evaluation use the existing ChipWhisperer environment.

**USB register contract**

Addresses use a 16-bit byte offset (target.bytecount_size=16). Register 0 exposes
the eight-byte signature RGBBAY01. Register 1 accepts control 1=start or 2=soft
reset. Register 2 holds width_log2 (5,6,7). Register 3 returns flags (busy,done,error),
16-bit little-endian count and RGB byte phase. Register 5 accepts RGB byte streams;
chunks may reset the offset and even split RGB triplets. Register 6 exposes the
captured Bayer bytes. The host finishes uploads before reads; live concurrent
read/write operation is outside the contract.
