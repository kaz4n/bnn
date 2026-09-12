#!/usr/bin/env bash
# Item 3 -- can a template be made tolerant to a change of clock?
#
# The 30 August result: a template profiled at 5 MHz scores F1 0.843 when the
# attack is also at 5 MHz, and 0.007 at 10 MHz (it returns an all-background
# image for every test image). Per-capture median/MAD normalisation is already
# in the feature path -- that is why a gain change costs nothing -- so it is not
# the missing piece. This tests the other option: profile under BOTH clocks.
#
# Design.
#   * profiling images 5300..5339, evaluation images 5200..5239. Disjoint sets,
#     so no image is ever profiled and attacked.
#   * every template is built from 40 profiling images in total. The mixed one
#     takes 20 images at 5 MHz and a DIFFERENT 20 at 10 MHz. Holding the total
#     fixed matters: the paper-scale run showed a bigger template at the same
#     delta returns more candidates per window and scores worse, so an unmatched
#     mixed template would confound condition-mixing with that effect.
#   * each template is attacked at both clocks, giving a 3x2 matrix.
set -u
PY="C:/Users/narut/ChipWhisperer/cwenv/Scripts/python.exe"
ROOT="C:/Users/narut/OneDrive/Desktop/Project/bnn"
HT="$ROOT/attack/hardtests_20260830"
cd "$ROOT/attack" || exit 1

P5="$ROOT/host/traces_prof_5m"
P10="$ROOT/host/traces_prof_10m"
for d in "$P5" "$P10"; do
  [ -d "$d" ] || { echo "missing profiling capture: $d"; exit 1; }
done

# --- templates, all 40 profiling images total -------------------------------
"$PY" active_power_template_cw305.py build-template --traces "$P5"  --profile-count 40 \
    --feature mean_abs --out "$HT/mix_tmpl_5m_p40.npz"  >/dev/null || exit 1
"$PY" active_power_template_cw305.py build-template --traces "$P10" --profile-count 40 \
    --feature mean_abs --out "$HT/mix_tmpl_10m_p40.npz" >/dev/null || exit 1
# The mix takes the first 20 profiling images at 5 MHz and the OTHER 20 at
# 10 MHz, so it still sees 40 distinct images -- same image diversity and same
# row count as each single-condition template it is compared against.
"$PY" "$HT/merge_templates.py" \
    --inputs "$HT/mix_tmpl_5m_p40.npz" "$HT/mix_tmpl_10m_p40.npz" \
    --image-slices 0:20 20:40 \
    --out "$HT/mix_tmpl_mixed_p40.npz" || exit 1
echo "templates built"

# --- evaluation bundles, same 40 images at each clock ------------------------
for C in C1_control C4_clk10m; do
  [ -d "$HT/mix_bundle_$C" ] || \
  "$PY" active_power_template_cw305.py make-bundle --traces "$ROOT/host/traces_hard_$C" \
      --start-index 0 --count 40 --out "$HT/mix_bundle_$C" \
      --truth-out "$HT/mix_truth_$C.json" >/dev/null || exit 1
done
echo "bundles built"

# --- 3 templates x 2 attack clocks ------------------------------------------
for T in 5m 10m mixed; do
  for C in C1_control C4_clk10m; do
    tag="${T}_vs_${C}"
    "$PY" active_power_template_cw305.py attack \
        --template "$HT/mix_tmpl_${T}_p40.npz" --bundle "$HT/mix_bundle_$C" \
        --out "$HT/mix_results_$tag" --feature mean_abs --delta 1.0 --group-size 3 \
        >/dev/null || exit 1
    "$PY" score_all.py --kind active --results "$HT/mix_results_$tag" \
        --truth "$HT/mix_truth_$C.json" --name "mix_$tag" \
        --out "$HT/score_mix_$tag.json" >/dev/null || exit 1
    echo "  $tag done"
  done
done
echo "MIXED CONDITION DONE"
