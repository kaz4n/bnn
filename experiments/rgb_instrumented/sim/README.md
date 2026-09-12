# Digital Bayer capture RTL simulation

These self-checking simulations exercise intentional digital RGB-to-RGGB sampling. They do not model analog measurements, USB electrical timing, or a physical hardware capture.

## Run

From the repository root in PowerShell:

```powershell
& './experiments/rgb_instrumented/sim/run_xsim.ps1'
& './experiments/rgb_instrumented/sim/run_wrapper_xsim.ps1'
```

Both runners accept `-VivadoBin` and use the existing Vivado 2016.4 installation by default. The wrapper runner also accepts `-Frontend` for the installed stock NewAE `cw305_usb_reg_fe.v`; it does not copy or modify that source. Generated files are confined to the ignored `sim/build/` directory. A complete PASS marker is required in addition to successful tool exit codes.

## Core interface and checks

`../rtl/rgb_bayer_capture.sv` accepts `rgb_pixel = {R,G,B}` in row-major order on each rising edge with `busy && rgb_valid`, provided neither `reset` nor `start` is asserted. Supported `width_log2` values are 5, 6, and 7 for square 32, 64, and 128 frames. A valid idle start latches the size and clears the count and error. Start has priority over a simultaneous input pixel.

Each accepted pixel stores one byte: red at even row/even column, blue at odd row/odd column, and green at the other two positions. The 15-bit sample count can represent the complete 16,384-byte largest frame. Completion clears busy and sets sticky done. Extra pixels set error without writing memory.

Every start clears done. A busy start is rejected, sets error, and preserves the active frame and count. An unsupported idle start sets error and preserves the stored frame and count. Reset synchronously clears status; a supported idle start also clears error.

The inferred 16,384-by-8-bit block RAM is not reset. Only addresses below the current sample count are valid frame data. An enabled read produces its byte after the next rising edge; a disabled read holds the output. Reset clears the read output register. Concurrent read/write of the same address is outside the interface contract.

`tb_rgb_bayer_capture.sv` independently walks row and column coordinates and checks every stored byte for all three frame sizes: 21,504 bytes total. It checks deterministic pseudo-random valid gaps, every unsupported size, width changes during capture, start priority, rejected busy starts, frame completion and count, extra pixels, read latency and enable hold, and reset during a partial frame. A watchdog fails stalled simulations.

## Wrapper checks

`tb_cw305_rgb_top.sv` instantiates `../hardware/cw305_rgb_top.sv`, the stock frontend, and a simulation-only functional BUFG stub. It checks the signature and width registers, unsupported widths, status/count/phase and LEDs, soft reset, and ignored offsets for the width register.

Its RGB stream keeps write strobes asserted while increasing byte offsets and holds each byte for one to five clocks. Chunk lengths of 1, 1, 511, 1,024, 17, and 256 bytes repeatedly restart the offset at zero and split RGB triplets. Both 32-by-32 and 128-by-128 frames are read back completely after an extra pixel is rejected: 17,408 bytes total. The test also resets a partial RGB triplet and has a watchdog.

All register and trace reads use the supplied minimum SMC profile: one address-setup clock followed by three clocks with NRD low; data is sampled immediately before NRD rises. Consecutive reads start the next address setup at that same sampling edge, giving exactly four clocks per transaction. The nominal 96 MHz clock rounds to a 10.416 ns period at the simulator's 1 ps resolution. The test checks the 41.664 ns transaction and 31.248 ns low-strobe durations and passes 18,169 reads, including 17,975 consecutive pairs. This replaces the earlier six-clock held-read test.

`wrapper_waveforms.tcl` records selected bus and pipeline signals in `build/wrapper/minimum_read_timing.vcd` and `rgb_wrapper_tb.wdb`. Frame arrays are excluded. `sampled_read_address` and `sampled_read_data` retain the sampled byte while `read_sample_toggle` marks each sampling edge, including when the next address changes on that same edge.

The final wrapper reads RAM from raw `usb_addr`. For the first captured-frame read, address setup starts at 102,139.296 ns. The recorded RTL waveform shows:

| Time after setup begins | Event |
| --- | --- |
| 0 ns | Address `0x60000`, NRD high |
| 5.208 ns | First rising edge loads RAM output `0x4A` |
| 10.416 ns | NRD goes low after one setup clock |
| 15.624 ns | Frontend enables the bus, which drives `0x4A` |
| 41.664 ns | Host samples `0x4A`; address changes to `0x60001` |
| 46.872 ns | Next rising edge loads and drives `0xAF` |
| 83.328 ns | Host samples `0xAF`; next four-clock read begins |

This proves the digital protocol at the modeled shared-clock phase. It does not establish physical input/output delays or independently justify a timing exception. A setup/hold exception requires separate validation of the actual launch/capture edges and external timing requirements.

## Handoff record

- Verified on 2026-09-11 with Vivado XSim 2016.4: core PASS (21,504 bytes, finish at 754,486 ns); final wrapper PASS (17,408 bytes using the minimum read profile, finish at 2,412,862.192 ns). The wrapper covered 10 chunks for 32-by-32 and 165 chunks for 128-by-128. Compilation and elaboration reported no RTL warnings. Final wrapper source SHA-256: `F829AD55BD48869214BCE2AC00499F37AEF31EC090C3D27385C26223CF48B7DA`.
- Objective: synthesizable intentional RGGB byte capture and independently checked core/wrapper behavior.
- Modified files: `../rtl/rgb_bayer_capture.sv`; both testbenches and runners in this directory; `bufg_sim_stub.sv`; `wrapper_waveforms.tcl`; `.gitignore`; this document. The parent owns all hardware-wrapper changes.
- Evidence: XSim compiler, elaboration, and run logs under `build/` (core) and `build/wrapper/` (wrapper). Each passing run prints the exact verified byte count.
- Limitations: behavioral RTL simulation; no post-route timing, electrical validation, or physical hardware results are implied. The wrapper uses a functional clock-buffer model.
- Next action: integrate these sources into the parent hardware flow and evaluate its separate synthesis and hardware evidence.
