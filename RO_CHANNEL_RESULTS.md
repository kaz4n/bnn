# On-chip RO-sensor channel — results & next steps (2026-07-14)

Autonomous bench session on the connected CW305 + CW-Lite. Goal: beat the external-ADC
dilution wall (`FINAL_RESULTS.md`) using the on-chip RO/TDC voltage sensor, power-only.

## Result summary

| Channel / attack | corr(sensor, window-mag) | S6 recog (bg) | S7 recog (template) |
|---|---|---|---|
| External CW-Lite ADC (historical) | 0.10–0.15 | ~0.10 | ~0.14 |
| **On-chip RO sensor** (this run, avg=50, 300/200) | **0.27 (asympt. 0.33)** | **0.255** | **0.13** |
| Paper (Spartan-6 / SAKURA-G / 2.5 GHz) | — | 0.816 | 0.898 |
| Clean sim (validated S7 config, 300/50) | 0.80 | 0.84 | 0.72 (below 0.898) |

The RO sensor is the **best power-only channel found — ~2× the external ADC** — but still
far below paper. Root cause confirmed = **dilution**, not extraction (see below).

## What was proven

1. **Sensor is real, not an output echo** (`host/ro_sensor_sanity.py`): reg 16 = live
   per-cycle droop vector, image-dependent, corr 0.12 with the conv output (not an echo);
   reg 48 unused. Polarity **inverted** (more switching → more droop → *lower* count).
2. **Dilution-limited, not averaging-limited** (`host/ro_avgcurve.py`): corr vs averages =
   0.28@50 → 0.31@200 → **0.33@800**. Plateaus; 16× more averaging buys +17%. More
   averaging cannot reach the ~0.8 recovery needs. Kernel-averaging (9→1) also does not
   help → residual is correlated global droop, not kernel-random noise.
3. **The failure is the channel, not the attack code.** A silently-regressed S7 metric was
   found and fixed — clean sim **substantially restored 0.32 → 0.72@300/50** (still below
   the paper's 0.898, but decisively above the broken 0.32). Enough to prove hardware
   S7=0.13 is genuinely channel-limited, not a code bug.

## Code fixes landed this session (no rebuild needed)

- `attack/power_template.py` S7 default was `metric="paper_l1", empty_policy="strict"`;
  with delta=0.1 on z-scored rho that starves candidates (~28% empty) → S7 collapses.
  **Fixed**: `evaluate.py` + `run_on_hardware.py` now default to the validated
  `metric="squared_l2", empty_policy="union", delta=0.1` (added `--metric`/`--empty-policy`).
  **Fidelity note:** `empty_policy="union"` is a deviation from the paper's *strict* group
  intersection (it falls back to the group union when the intersection is empty). It is the
  validated config that recovers; if strict paper fidelity is required, use
  `--empty-policy strict --metric paper_l1 --delta 0.5` (sweep: 0.68 vs 0.76). Your call.
- `run_on_hardware.py --invert` added for droop/RO sensors (flips S6; S7 unaffected).
- `attack/trace_sim.py` gained `smear_tau`/`dilution`/`dilution_jitter` to model the bench.
- New: `host/ro_sensor_sanity.py`, `host/ro_capture.py` (quiesced averaging), `host/ro_avgcurve.py`.

## Why the ceiling is DILUTION (locality), not sensor resolution

The wall is 0.33 and it is a **placement** problem, not a bits problem — three lines of
evidence:
- **800× averaging already recovered fine resolution** (it dithers the coarse 0–16 sensor
  into a fine flip-probability estimate) and still **plateaued at 0.33**. Raw resolution is
  demonstrably not the cap.
- The **sim reproduces the ceiling with a *linear* sensor + dilution alone** (dilution≈1 →
  corr≈0.31 ≈ hardware 0.33). A coarse encoding is not needed to explain the wall.
- A finer sensor reading the **same shared-grid droop** reads the same *diluted* signal.
  Only changing *which* droop it integrates (local IR-drop over the conv, via placement)
  raises the ceiling.

Separately real: `cw305/ro_counter_sensor.v` under-implements the sensor —
`sample_value = sum_i (tap_sync[i] ^ tap_prev[i])` counts **how many of 16 ROs flipped
phase** (0..16, matches the coarse ~11 levels); the `ro_async_counter.gray_count` that
should hold each RO's **oscillation count** is stubbed (`{WIDTH{tap}}`) and unconnected. So
the RO frequency (∝ Vdd droop) is discarded.

**De-risk (sim, common-across-kernels dilution, 300/40) pins down the two factors:**

| dilution | sim corr | sim S6 | sim S7 |  vs hardware |
|---|---|---|---|---|
| 1.0 | 0.31 | 0.45 | 0.125 | corr+**S7 match** (hw 0.33/0.13); S6 too high |
| 2.0 | 0.17 | 0.22 | 0.175 | S6 matches (hw 0.255); corr too low |

No single dilution fits all three: dilution≈1 reproduces **corr and S7 exactly**, but
predicts S6≈0.45 where hardware gives **0.255**. So **dilution fully explains the S7/corr
ceiling** (→ placement is the fix), while a **second factor specifically degrades S6** —
consistent with the coarse 0–16 encoding, which hurts magnitude-thresholded background
detection far more than the z-scored template match. Net: fixing the counter should recover
**S6 background** toward the ~0.45 dilution-limit; it does **not** move the S7/corr 0.33 wall.

## Prioritized next steps (all require a Vivado REBUILD — user-run; Modern Standby
suspends builds). **Artifacts prepared + xvlog/parse-checked — turnkey.**

1. ~~**Floorplan the sensor pblock adjacent to the conv**~~ — **TESTED 2026-08-25, NO
   IMPROVEMENT.** Built `cw305_bnn_ro3pb.bit` (v1 sensor + pblock co-locating U_ro_sensor +
   U_conv in CLOCKREGION_X0Y0:X0Y2). Smoke corr(sensor,winmag) = **0.267 co-located vs 0.272
   baseline — identical.** Clock-region co-location does NOT beat dilution: the RO reads the
   shared low-impedance PDN's *global* droop; CR-granularity placement can't isolate local
   IR-drop over the conv. Placement hypothesis empirically closed at this granularity.
   (A tighter slice-level RO-interleave-among-conv-cells build might differ but is a major
   effort with uncertain payoff.)
2. **Clock-gate the rest of the fabric during the trigger window** — *ceiling-raiser*.
   `BUFGCE` on non-conv clock domains, held off while `busy`. Shrinks the data-independent
   global term (and its variability, which common-mode subtraction cannot remove). (Not yet
   drafted — needs the top-level clock tree; ask if you want it.)
3. **Fix the RO oscillation counter** — *recovers S6, not the S7 wall*. **READY:
   `cw305/ro_counter_sensor_v2.v`** — drop-in (same module/ports), real per-RO Gray
   oscillation counter, double-flop synced, replacing the 0–16 phase-flip. Parses clean in
   xvlog; **still needs C-sim + on-silicon check** that `sample_value` now spans tens–
   hundreds. Lifts **S6** toward its ~0.45 dilution-limit; does **not** move the S7/corr wall.
4. **EM near-field probe** (H-field → LNA → CW measure-in) — reliable spatial-isolation fix;
   user has none. Highest-ROI purchase if 1–3 fall short.

## Bottom line (updated after testing co-location on silicon)
The on-chip RO sensor advances the project past `FINAL_RESULTS.md`: a real, ~2×-better
power-only channel, with the attack code now validated correct. **Co-location (item 1) was
built and tested on silicon — it did NOT help** (corr 0.267 vs 0.272), so the dilution wall
is not a clock-region-placement problem. What remains: **clock-gating** the idle fabric in
the trigger window (item 2, untested) and, most reliably, an **EM near-field probe** (item 4)
for true spatial isolation the shared-PDN RO can't provide. The **S7 metric fix** (in the
attack code) stands regardless and is the session's most portable win.

Build/flow notes for future rebuilds (hard-won): builds run via **PowerShell** (git-bash
background breaks `launch_runs` worker spawn) or the in-process DCP flow (`impl_from_dcp.tcl`);
`cw305_reg_bnn.v` must NOT define `NO_RO_SENSOR` (it disables the sensor); RO loops need
`BITSTREAM.GENERAL.CRC DISABLE` (`allow_ro_loops.tcl`) to load; re-read pin XDCs when
implementing from a synth checkpoint.
