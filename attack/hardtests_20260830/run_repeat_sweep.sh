#!/usr/bin/env bash
# H1 -- averaging sweep. One avg=5 capture, subset post-hoc to R=1..5.
# Profile and attack both use the same R, so the curve answers
# "how many hardware repeats does the attack actually need".
set -u
PY="C:/Users/narut/ChipWhisperer/cwenv/Scripts/python.exe"
ROOT="C:/Users/narut/OneDrive/Desktop/Project/bnn"
SRC="$ROOT/host/traces_rerun_20260830_trained_80"
HT="$ROOT/attack/hardtests_20260830"
cd "$ROOT/attack" || exit 1

for R in 1 2 3 4 5; do
  echo "=============== R=$R ==============="
  TR="$ROOT/host/traces_sweep_R${R}"
  if [ "$R" -eq 5 ]; then TR="$SRC"; else
    "$PY" "$HT/make_repeat_subset.py" --src "$SRC" --dst "$TR" --repeats "$R" >/dev/null || exit 1
  fi
  "$PY" active_power_template_cw305.py build-template \
      --traces "$TR" --profile-count 40 --feature mean_abs \
      --out "$HT/sweep_profile_R${R}.npz" >/dev/null || exit 1
  "$PY" active_power_template_cw305.py make-bundle \
      --traces "$TR" --start-index 820 --count 40 \
      --out "$HT/sweep_bundle_R${R}" --truth-out "$HT/sweep_truth_R${R}.json" >/dev/null || exit 1
  "$PY" active_power_template_cw305.py attack \
      --template "$HT/sweep_profile_R${R}.npz" --bundle "$HT/sweep_bundle_R${R}" \
      --out "$HT/sweep_results_R${R}" --feature mean_abs --delta 1.0 --group-size 3 >/dev/null || exit 1
  "$PY" score_all.py --kind active --results "$HT/sweep_results_R${R}" \
      --truth "$HT/sweep_truth_R${R}.json" --name "sweep_R${R}" \
      --out "$HT/score_sweep_R${R}.json" >/dev/null || exit 1
  echo "R=$R done"
  # reclaim the derived trace dir immediately; it is regenerable from SRC
  if [ "$R" -ne 5 ]; then rm -rf "$TR"; fi
done
echo "SWEEP DONE"
