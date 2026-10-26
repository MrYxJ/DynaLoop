#!/usr/bin/env bash
# Reproduce every headline number quoted in README.md.
#
# Everything is written to ``results/`` as JSON; a cell whose output file
# already exists is skipped, so the script is safe to re-run after a crash
# (this machine's GPU driver occasionally dies, hence the per-cell files).
#
# Usage:
#   export DYNALOOP_MODEL=/path/to/Ouro-1.4B
#   bash scripts/repro_results.sh
#
# Optional filters:
#   CELLS="niah_full niah_dl" bash scripts/repro_results.sh
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PYTHON:-python}"
export PYTHONPATH="$ROOT${PYTHONPATH:+;$PYTHONPATH}"
export PYTHONIOENCODING=utf-8
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
MODEL="${DYNALOOP_MODEL:-Ouro-1.4B}"

N=100
mkdir -p "$ROOT/logs" "$ROOT/results"

# The shared budget rule used by every DynaLoop cell:
#   r = clip(kappa * (JS(s_{t-1,l}, s_{t,l}) / D1)^gamma, floor, 1)
DL_ARGS="--mode dynaloop --allocator js --keep-loops 1 --block 1 \
--chunk 5 --bmin 0.02 --min-prompt 192 --js-floor 0.10"

done_file () { ls "$ROOT/results"/$1 2>/dev/null | head -1; }

echo "=== DynaLoop repro starting $(date '+%F %T') model=$MODEL"

# ------------------------------------------------------------------ #
# 1. per-(loop, layer) budget heatmap (figure data)                   #
# ------------------------------------------------------------------ #
if [ -z "$(done_file 'budget_heatmap.json')" ]; then
  echo "--- heatmap $(date '+%F %T')"
  "$PY" -X utf8 "$ROOT/examples/budget_heatmap.py" --model "$MODEL" \
      --out "$ROOT/results/budget_heatmap" \
      > "$ROOT/logs/heatmap.log" 2>&1
  echo "    rc=$? $(date '+%F %T')"
fi


# --- shared argv ----------------------------------------------------- #
# lm-eval cells: n=100, ~1k-token few-shot prompt, 512 generated tokens.
MATH_COMMON="--n $N --seed 42 --target-prompt 1000 --batch 4 \
--gen-toks 512 --max-gen-toks 512"

# ------------------------------------------------------------------ #
# 2. NIAH-Multi (1024 ctx, 16 needles, n=100)                         #
# ------------------------------------------------------------------ #
NIAH="--n $N --batch 4 --length 1024 --needles 16"

if [ -z "$(done_file '*fullkv_n100_niahm.json')" ]; then
  echo "--- niah fullkv $(date '+%F %T')"
  "$PY" -X utf8 "$ROOT/scripts/run_niah.py" --model "$MODEL" \
      --mode full $NIAH --tag fullkv_n100 \
      > "$ROOT/logs/niah_fullkv.log" 2>&1
  echo "    rc=$? $(date '+%F %T')"
fi

if [ -z "$(done_file '*dynamickv_r055_n100_niahm.json')" ]; then
  echo "--- niah dynamickv $(date '+%F %T')"
  "$PY" -X utf8 "$ROOT/scripts/run_niah.py" --model "$MODEL" \
      --mode dynaloop --allocator block --keep-loops 0 --block 6 \
      --chunk 5 --bmin 0.05 --min-prompt 192 --rho 0.55 $NIAH \
      --tag dynamickv_r055_n100 \
      > "$ROOT/logs/niah_dkv.log" 2>&1
  echo "    rc=$? $(date '+%F %T')"
fi

if [ -z "$(done_file '*dynaloop_k050_n100_niahm.json')" ]; then
  echo "--- niah dynaloop $(date '+%F %T')"
  "$PY" -X utf8 "$ROOT/scripts/run_niah.py" --model "$MODEL" \
      $DL_ARGS --js-kappa 0.5 --js-gamma 1.0 $NIAH \
      --tag dynaloop_k050_n100 \
      > "$ROOT/logs/niah_dynaloop.log" 2>&1
  echo "    rc=$? $(date '+%F %T')"
fi

# ------------------------------------------------------------------ #
# 3. GSM8K / MATH-500 through lm-eval                                 #
# ------------------------------------------------------------------ #
run_math () {   # $1=tag  $2=task  $3..=args
  local tag="$1"; local task="$2"; shift 2
  if [ -n "$(done_file "lmeval_*_${tag}.json")" ]; then
    echo "=== $tag/$task already done"; return 0
  fi
  echo "--- $tag/$task $(date '+%F %T')"
  "$PY" -X utf8 "$ROOT/scripts/run_math.py" --model "$MODEL" \
      --task "$task" $MATH_COMMON --tag "$tag" "$@" \
      > "$ROOT/logs/${tag}_${task}.log" 2>&1
  echo "    rc=$? $(date '+%F %T')"
}

# baselines
run_math "s15_full_gsm8k"  gsm8k   --mode full
run_math "s15_dkv_gsm8k"   gsm8k   --mode dynamickv --rho 0.55
run_math "s16_full_math500" math500 --mode full
run_math "s16_dkv_math500"  math500 --mode dynamickv --rho 0.55
# DynaLoop kappa sweep
run_math "s22_k25_math500"  math500 $DL_ARGS --js-kappa 0.25 --js-gamma 1.0
run_math "s21_k50_gsm8k"    gsm8k   $DL_ARGS --js-kappa 0.5  --js-gamma 1.0
run_math "s21_k50_math500"  math500 $DL_ARGS --js-kappa 0.5  --js-gamma 1.0

echo "ALLDONE $(date '+%F %T')" > "$ROOT/logs/repro_ALLDONE.txt"
echo "$?" > /dev/null
tail -1 "$ROOT/logs/repro_ALLDONE.txt"
