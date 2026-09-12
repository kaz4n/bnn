#!/usr/bin/env bash
# Extend the Section 8.5 mixing sweep below its floor.
#
# 8.5 swept delta over {0.10, 0.15, 0.25, 0.40}. The mixed template's best value
# sat at 0.10, the bottom of that range, so its optimum was never bracketed --
# the same flaw being corrected in 8.3. My own stated mechanism (wrong-condition
# rows survive the distance test, so tightening delta helps mixing) predicts the
# mixed template keeps improving below 0.10. Untested prediction; test it.
#
# 3 templates x 2 bundles x 4 radii = 24 runs. No board needed.
set -u
PY="C:/Users/narut/ChipWhisperer/cwenv/Scripts/python.exe"
ROOT="C:/Users/narut/OneDrive/Desktop/Project/bnn"
HT="$ROOT/attack/hardtests_20260830"
cd "$ROOT/attack" || exit 1

TMPLS="5m 10m mixed"
BUNDLES="5m 10m"
DELTAS="0.02 0.03 0.04 0.06"

for T in $TMPLS; do
  for B in $BUNDLES; do
    for D in $DELTAS; do
      DT=${D/./p}
      S="$HT/score_mx2_${T}_vs_${B}_d${DT}.json"
      [ -f "$S" ] && continue
      "$PY" active_power_template_cw305.py attack \
          --template "$HT/mx2_tmpl_${T}.npz" --bundle "$HT/g4_bundle_${B}" \
          --out "$HT/tmp_mx2_${T}_vs_${B}_d${DT}" --feature mean_abs --delta "$D" \
          --group-size 3 >/dev/null 2>&1 || { echo "  ${T}->${B} d=${D} ATTACK FAILED"; continue; }
      "$PY" score_all.py --kind active --results "$HT/tmp_mx2_${T}_vs_${B}_d${DT}" \
          --truth "$HT/g4_truth_${B}.json" --name "mx2_${T}_vs_${B}_d${DT}" \
          --out "$S" >/dev/null 2>&1 || { echo "  ${T}->${B} d=${D} SCORE FAILED"; continue; }
      rm -rf "$HT/tmp_mx2_${T}_vs_${B}_d${DT}"
    done
    echo "  ${T} -> ${B} swept low"
  done
done
echo "MIX2 LOW-DELTA SWEEP DONE"
