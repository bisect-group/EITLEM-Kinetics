#!/usr/bin/env bash
# Reproduce EITLEM-Kinetics: ESM1v cache, train and test kcat and km (all splits, seeds), seed summary.
# Usage: emulator_bench/run.sh [--dry-run] [cache|train|aggregate|all]
# Env: PY, GPUS, RUNS_PER_GPU, SEEDS, NUM_WORKERS, DATA_ROOT, OUT_ROOT, DRY_RUN=1.
# train also evaluates (final test metrics); finished runs are skipped.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

PY=${PY:-python}
read -ra GPUS <<<"${GPUS:-0}"
read -ra SEEDS <<<"${SEEDS:-666 777 888}"
RUNS_PER_GPU=${RUNS_PER_GPU:-1}
NUM_WORKERS=${NUM_WORKERS:-4}
DATA=() OUT=() EMB=()
[[ -z ${DATA_ROOT:-} ]] || DATA=(--data_root "$DATA_ROOT")
[[ -z ${OUT_ROOT:-} ]] || { OUT=(--output_dir "$OUT_ROOT"); EMB=(--embeddings_dir "$OUT_ROOT/embeddings"); }
STEP=all
for a in "$@"; do
  case $a in
    --dry-run) DRY_RUN=1 ;;
    cache | train | aggregate | all) STEP=$a ;;
    *) echo "usage: $0 [--dry-run] [cache|train|aggregate|all]" >&2; exit 2 ;;
  esac
done
run() { echo "+ $*"; [[ -n ${DRY_RUN:-} ]] || "$@"; }
want() { [[ $STEP == all || $STEP == "$1" ]]; }

if want cache; then
  run env CUDA_VISIBLE_DEVICES="${GPUS[0]}" "$PY" emulator_bench/cache_embeddings.py --device cuda:0 "${DATA[@]}" "${EMB[@]}"
fi
if want train; then
  run "$PY" emulator_bench/run_split_benchmarks.py --skip_cache --gpus "${GPUS[@]}" --runs_per_gpu "$RUNS_PER_GPU" \
    --num_workers "$NUM_WORKERS" --seeds "${SEEDS[@]}" "${DATA[@]}" "${OUT[@]}"
fi
if want aggregate; then
  run "$PY" emulator_bench/aggregate_eitlem_results.py "${OUT[@]}"
fi
