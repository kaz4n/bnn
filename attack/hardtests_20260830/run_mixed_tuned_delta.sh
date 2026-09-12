#!/usr/bin/env bash
# Re-run the clock-transfer matrix at a TUNED search radius.
#
# Two reasons this is not just a repeat.
#
# 1. The delta sweep showed delta=1.0 (the value carried over from the paper) is
#    far from optimal here: the 40-image template goes from F1 0.843 to 0.956 at
#    delta=0.3. Every conclusion drawn at delta=1.0 deserves a re-check.
# 2. My explanation for why the MIXED template loses to the pure 10 MHz one was
#    that its 5 MHz rows contribute wrong candidates that still fall inside
#    delta. That explanation predicts mixing should improve as delta shrinks.
#    If it does not, the explanation is wrong.
set -u
PY="C:/Users/narut/ChipWhisperer/cwenv/Scripts/python.exe"
ROOT="C:/Users/narut/OneDrive/Desktop/Project/bnn"
HT="$ROOT/attack/hardtests_20260830"
DELTA="${1:-0.3}"
DTAG="${DELTA/./p}"
cd "$ROOT/attack" || exit 1

for C in C1_control C4_clk10m; do
  if [ ! -d "$HT/mix_bundle_$C" ]; then
    "$PY" active_power_template_cw305.py make-bundle \
        --traces "$ROOT/host/traces_hard_$C" --start-index 0 --count 40 \
        --out "$HT/mix_bundle_$C" --truth-out "$HT/mix_truth_$C.json" >/dev/null || exit 1
  fi
done
echo "bundles ready (delta=$DELTA)"

for T in 5m 10m mixed; do
  for C in C1_control C4_clk10m; do
    tag="${T}_vs_${C}_d${DTAG}"
    score="$HT/score_mixd_${tag}.json"
    [ -f "$score" ] && { echo "  $tag cached"; continue; }
    "$PY" active_power_template_cw305.py attack \
        --template "$HT/mix_tmpl_${T}_p40.npz" --bundle "$HT/mix_bundle_$C" \
        --out "$HT/mixd_results_$tag" --feature mean_abs --delta "$DELTA" \
        --group-size 3 >/dev/null 2>&1 || { echo "  $tag ATTACK FAILED"; continue; }
    "$PY" score_all.py --kind active --results "$HT/mixd_results_$tag" \
        --truth "$HT/mix_truth_$C.json" --name "mixd_$tag" \
        --out "$score" >/dev/null 2>&1 || { echo "  $tag SCORE FAILED"; continue; }
    rm -rf "$HT/mixd_results_$tag"
    echo "  $tag done"
  done
done
echo "MIXED TUNED-DELTA DONE"
