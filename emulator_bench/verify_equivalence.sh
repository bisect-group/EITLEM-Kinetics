#!/usr/bin/env bash
# Checks that the emulator_bench wrapper reproduces the original EITLEM-Kinetics code path.
# Usage: verify_equivalence.sh [--gpu | --negative-control]
# Expected env: mldb (torch, torch_geometric, rdkit, pandas, pytest; Biohub `esm`). Never `pip install fair-esm`: it overwrites `esm` in the env.
# The tests get fair-esm 2.0.0 from emulator_bench/esm_deps.py (setup_esm). It unpacks the sha256-checked wheel into
# <out_root>/pydeps and never changes site-packages. The first run needs network once unless that folder exists.
# This script does not create envs.
set -euo pipefail
cd "$(dirname "$0")/.."
echo "Expected env: mldb (Biohub esm; fair-esm comes from esm_deps.py, do not pip install it)"

gpu=0
show=""
markers="not gpu and not esm_weights"
if [[ "${1:-}" == "--gpu" ]]; then
    gpu=1
    markers="not esm_weights"
    show="-s"
    echo "GPU mode: also runs the side-stream loader and TF32 tests. They skip when no CUDA device is visible."
elif [[ "${1:-}" == "--negative-control" ]]; then
    # Scales the production run's initial weights by 1.05. Every precision test must FAIL, else its bound is vacuous.
    out=$(EQUIV_NEGATIVE_CONTROL=1 python -m pytest emulator_bench/tests/test_original_equivalence.py -q -p no:cacheprovider --color=no -s -m gpu -k within_fp32_noise_control 2>&1) || true
    echo "$out"
    if grep -Eq "[0-9]+ passed" <<<"$out" || ! grep -Eq "[0-9]+ failed" <<<"$out"; then
        echo "FAIL: a GPU test did not fail under the negative control"
        exit 1
    fi
    echo "OK: every GPU precision test fails under the negative control"
    exit 0
else
    export CUDA_VISIBLE_DEVICES=""
fi

out=$(python -m pytest emulator_bench/tests/test_original_equivalence.py -q -rs -p no:cacheprovider --color=no $show -m "$markers" 2>&1) || { echo "$out"; exit 1; }
echo "$out"
if grep -Eq "[0-9]+ skipped" <<<"$out"; then
    if [[ $gpu -eq 1 ]] && ! python -c "import torch; raise SystemExit(not torch.cuda.is_available())"; then
        echo "No CUDA device is visible: the GPU tests were skipped."
    else
        echo "FAIL: tests were skipped"
        exit 1
    fi
fi
echo "OK: wrapper matches the original code path"
