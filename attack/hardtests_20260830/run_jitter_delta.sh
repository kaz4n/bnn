#!/usr/bin/env bash
# Delta sweep for the clock-randomised evaluation set.
#
# Section 8.4 compares each template against the jitter set at a single delta = 0.25.
# The per-cell sweep (run_grid_delta.sh) retunes the matched-clock column, so leaving
# the randomised column at 0.25 would compare a tuned number against an untuned one --
# the exact error being corrected in 8.3. Sweep both sides on the same radii.
#
# 4 templates x 6 radii = 24 runs. No board needed.
set -u
PY="C:/Users/narut/ChipWhisperer/cwenv/Scripts/python.exe"
ROOT="C:/Users/narut/OneDrive/Desktop/Project/bnn"
HT="$ROOT/attack/hardtests_20260830"
cd "$ROOT/attack" || exit 1

FREQS="5m 7p5m 10m 20m"
DELTAS="0.06 0.10 0.15 0.25 0.40 0.60"

[ -d "$HT/g4_bundle_jitter" ] || { echo "missing g4_bundle_jitter"; exit 1; }

for P in $FREQS; do
  for D in $DELTAS; do
    DT=${D/./p}
    S="$HT/score_jd_${P}_vs_jitter_d${DT}.json"
    [ -f "$S" ] && continue
    "$PY" active_power_template_cw305.py attack \
        --template "$HT/g4_tmpl_${P}.npz" --bundle "$HT/g4_bundle_jitter" \
        --out "$HT/jd_${P}_${DT}" --feature mean_abs --delta "$D" \
        --group-size 3 >/dev/null 2>&1 || { echo "  ${P} d=${D} ATTACK FAILED"; continue; }
    "$PY" score_all.py --kind active --results "$HT/jd_${P}_${DT}" \
        --truth "$HT/g4_truth_jitter.json" --name "jd_${P}_${D}" \
        --out "$S" >/dev/null 2>&1 || { echo "  ${P} d=${D} SCORE FAILED"; continue; }
    rm -rf "$HT/jd_${P}_${DT}"
  done
  echo "  jitter sweep done for template ${P}"
done
echo "JITTER DELTA SWEEP DONE"
