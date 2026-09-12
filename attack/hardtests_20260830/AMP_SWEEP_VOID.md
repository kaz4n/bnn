# Amplifier sweep of 2026-09-06 is VOID — do not use these numbers

`traces_gfash_l63`, `traces_gfash_l15`, `traces_gfash_l3`, `traces_gfash_l0` and the
`p2p_gfash_l*` results derived from them are all captures of the SAME bitstream
(LEAK_LANES=63). The per-point reprogramming silently failed.

Evidence — mean absolute trace value, which must fall with the amplifier width:

| LANES | amp flops | mean abs trace | expected |
|-------|-----------|----------------|----------|
| 63    | 4536      | 7471           | highest  |
| 15    | 1080      | 7478           | lower    |
| 3     |  216      | 7486           | lower    |
| 0     |    0      | 7500           | lowest   |

The spread is 0.4 % and the order is backwards. A design with 4536 dedicated
toggling flops and one with none cannot produce the same trace amplitude. All four
also share an identical functional signature (hw_min -778, hw_max 462), which four
different designs would not.

Cause: reprogramming the CW305 over an already-configured FPGA silently fails on
this board (see memory note `hw-condition-transfer`). `cw305_p2p_capture.py`
programs and then captures without verifying the new design is live, so it
happily captured the previously loaded bitstream four times.

## To redo this properly

The four bitstreams are grey designs with identical register maps, so neither the
AES page signature nor the byte-ramp probe can tell them apart. Amplitude is the
only discriminator, and it can only be checked after capturing.

Sequence per point, requiring a human:
1. Power-cycle the CW305 (cut power; USB RST is not enough).
2. Program `build/cw305_leakage_grey_d8_l<L>.bit` while blank.
3. Capture a handful of images and record mean abs trace BEFORE the full run.
4. Reject the point if the amplitude matches the previous LANES value.

The bitstreams themselves are fine and verified: register deltas over LANES=0 are
+215 / +1078 / +4536 for LANES 3 / 15 / 63, matching 72*LANES exactly, and all four
meet timing with 0 failing endpoints. Only the captures are void.
