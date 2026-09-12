#!/usr/bin/env bash
# Gap 3 -- the search radius has to be tuned with the template size.
#
# The 30 August paper-scale run showed a 300-image template scoring WORSE than a
# 40-image one (F1 0.759 against 0.843) while delta stayed at the published 1.0.
# Candidates per window doubled (14.1 -> 32.0) while the fallback count fell
# (1.00 -> 0.17), so coverage improved and selectivity got worse. This sweeps
# delta for both template sizes against the same 40 evaluation images, which is
# the cheapest way to turn that negative result into a tuning rule.
set -u
PY="C:/Users/narut/ChipWhisperer/cwenv/Scripts/python.exe"
ROOT="C:/Users/narut/OneDrive/Desktop/Project/bnn"
HT="$ROOT/attack/hardtests_20260830"
cd "$ROOT/attack" || exit 1

BUNDLE="$HT/delta_bundle_ctrl40"
TRUTH="$HT/delta_truth_ctrl40.json"
if [ ! -d "$BUNDLE" ]; then
  "$PY" active_power_template_cw305.py make-bundle \
      --traces "$ROOT/host/traces_hard_C1_control" --start-index 0 --count 40 \
      --out "$BUNDLE" --truth-out "$TRUTH" >/dev/null || exit 1
fi
echo "bundle ready"

run_one() {          # $1 = template tag, $2 = template path, $3 = delta
  local tag=$1 tmpl=$2 d=$3
  local dtag=${d/./p}
  local out="$HT/delta_results_${tag}_d${dtag}"
  local score="$HT/score_delta_${tag}_d${dtag}.json"
  [ -f "$score" ] && { echo "  ${tag} delta=${d} cached"; return 0; }
  "$PY" active_power_template_cw305.py attack \
      --template "$tmpl" --bundle "$BUNDLE" --out "$out" \
      --feature mean_abs --delta "$d" --group-size 3 >/dev/null 2>&1 || return 1
  "$PY" score_all.py --kind active --results "$out" --truth "$TRUTH" \
      --name "delta_${tag}_${d}" --out "$score" >/dev/null 2>&1 || return 1
  # the recovered images are regenerable from the template; keep only the score
  rm -rf "$out"
  echo "  ${tag} delta=${d} done"
}

for D in 0.08 0.12 0.15 0.25 0.2 0.3 0.4 0.6 0.8 1.0 1.4; do
  echo "=== delta $D ==="
  run_one p40  "$ROOT/attack/rerun_20260830/profile_trained_80_p40.npz" "$D" || echo "  p40 FAILED"
  run_one p300 "$HT/paper_profile_p300.npz" "$D"                        || echo "  p300 FAILED"
done
echo "DELTA SWEEP DONE"
