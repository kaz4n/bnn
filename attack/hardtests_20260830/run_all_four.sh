#!/usr/bin/env bash
# The four follow-up experiments, in one pass so they do not fight for CPU.
#
#  A. Four-frequency transfer grid   -- 4 profiling clocks x 4 attack clocks
#  B. Clock randomisation as defence -- the same templates against a trace set
#                                       whose clock changes per image
#  C. Mixing done properly           -- equal per-condition coverage this time,
#                                       delta swept because the mixed template is
#                                       twice the size of the ones it is compared to
#
# The wide-bitstream experiment (quantisation confound) is not here: it needs the
# board reprogrammed, which needs a power cycle.
#
# delta = 0.25 is the tuned optimum for a 40-image template on this hardware
# (13-point sweep, 1 Sept). The published 1.0 is not used anywhere.
set -u
PY="C:/Users/narut/ChipWhisperer/cwenv/Scripts/python.exe"
ROOT="C:/Users/narut/OneDrive/Desktop/Project/bnn"
HT="$ROOT/attack/hardtests_20260830"
H="$ROOT/host"
cd "$ROOT/attack" || exit 1

FREQS="5m 7p5m 10m 20m"
DELTA40=0.25

# ---------- templates, one per profiling clock -------------------------------
for F in $FREQS; do
  T="$HT/g4_tmpl_${F}.npz"
  [ -f "$T" ] || "$PY" active_power_template_cw305.py build-template \
      --traces "$H/traces_g4p_${F}" --profile-count 40 --feature mean_abs \
      --out "$T" >/dev/null || exit 1
done
echo "grid templates built"

# ---------- evaluation bundles, one per attack clock plus the jittered set ----
for F in $FREQS jitter; do
  B="$HT/g4_bundle_${F}"
  [ -d "$B" ] || "$PY" active_power_template_cw305.py make-bundle \
      --traces "$H/traces_g4e_${F}" --start-index 0 --count 40 \
      --out "$B" --truth-out "$HT/g4_truth_${F}.json" >/dev/null || exit 1
done
echo "grid bundles built"

attack_score() {   # $1 template path, $2 bundle tag, $3 delta, $4 score name
  local tmpl=$1 btag=$2 d=$3 name=$4
  local S="$HT/score_${name}.json"
  [ -f "$S" ] && { echo "  $name cached"; return 0; }
  "$PY" active_power_template_cw305.py attack --template "$tmpl" \
      --bundle "$HT/g4_bundle_${btag}" --out "$HT/tmp_${name}" \
      --feature mean_abs --delta "$d" --group-size 3 >/dev/null 2>&1 \
      || { echo "  $name ATTACK FAILED"; return 1; }
  "$PY" score_all.py --kind active --results "$HT/tmp_${name}" \
      --truth "$HT/g4_truth_${btag}.json" --name "$name" --out "$S" >/dev/null 2>&1 \
      || { echo "  $name SCORE FAILED"; return 1; }
  rm -rf "$HT/tmp_${name}"
  echo "  $name done"
}

# ---------- A. the 4x4 transfer grid -----------------------------------------
echo "=== A: four-frequency transfer grid ==="
for P in $FREQS; do
  for A in $FREQS; do
    attack_score "$HT/g4_tmpl_${P}.npz" "$A" "$DELTA40" "g4_${P}_vs_${A}"
  done
done

# ---------- B. clock randomisation as a countermeasure -----------------------
echo "=== B: clock randomisation as a defence ==="
for P in $FREQS; do
  attack_score "$HT/g4_tmpl_${P}.npz" "jitter" "$DELTA40" "g4_${P}_vs_jitter"
done

# ---------- C. mixing with equal per-condition coverage ----------------------
echo "=== C: mixing with equal per-condition coverage ==="
# pure templates come from the interleaved 2-condition capture; the mixed one is
# the union, so it holds 40 images at EACH clock rather than 20, and is twice the
# size of the templates it is compared against -- hence its own delta sweep.
for F in 5m 10m; do
  T="$HT/mx2_tmpl_${F}.npz"
  [ -f "$T" ] || "$PY" active_power_template_cw305.py build-template \
      --traces "$H/traces_ilv_${F}" --profile-count 40 --feature mean_abs \
      --out "$T" >/dev/null || exit 1
done
[ -f "$HT/mx2_tmpl_mixed.npz" ] || "$PY" "$HT/merge_templates.py" \
    --inputs "$HT/mx2_tmpl_5m.npz" "$HT/mx2_tmpl_10m.npz" \
    --out "$HT/mx2_tmpl_mixed.npz" >/dev/null || exit 1

for D in 0.10 0.15 0.25 0.40; do
  DT=${D/./p}
  for T in 5m 10m mixed; do
    for A in 5m 10m; do
      attack_score "$HT/mx2_tmpl_${T}.npz" "$A" "$D" "mx2_${T}_vs_${A}_d${DT}"
    done
  done
done

echo "ALL FOUR DONE"
