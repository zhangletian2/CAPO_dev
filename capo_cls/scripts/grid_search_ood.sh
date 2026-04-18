#!/bin/bash
# >>> capo/scripts/_paths.sh (auto-inserted)
# Locate scripts/_paths.sh relative to this script's location so the file
# stays portable after a git clone -- no absolute paths.
_CAPO_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
source "${_CAPO_SCRIPT_DIR}/../../scripts/_paths.sh"
# <<< capo/scripts/_paths.sh
# OOD grid search: tune on ImageNet val (1/10 subset = 5000 images).
# Fixed p=0.1, sweep tau × beta at 0.1 resolution.
# 70 combos × 5000 imgs × ~0.5s/img ÷ 8 GPUs ≈ 6h
#
# Usage: bash grid_search_ood.sh <gpu_list>
# Example: bash grid_search_ood.sh "0,1,2,3,4,5,6,7"

GPU_LIST=(${1:-"0,1,2,3,4,5,6,7"})
IFS=',' read -ra GPUS <<< "${GPU_LIST}"
NUM_GPUS=${#GPUS[@]}

MAX_SAMPLES=5000
BS=32
P=0.1

TAU_LIST=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0)
BETA_LIST=(0.3 0.5 0.7 0.8 0.9 1.0 1.1 1.3)

LOG_DIR=./logs/grid_ood_val
mkdir -p ${LOG_DIR}

total=$((${#TAU_LIST[@]} * ${#BETA_LIST[@]}))
echo "=== OOD Grid Search on ImageNet val (1/10 subset) ==="
echo "Fixed p=${P}, Combos: ${total}, GPUs: ${NUM_GPUS}, Samples: ${MAX_SAMPLES}, BS: ${BS}"
echo ""

job_idx=0
for tau in "${TAU_LIST[@]}"; do
  for beta in "${BETA_LIST[@]}"; do
    gpu=${GPUS[$((job_idx % NUM_GPUS))]}
    echo "[Job $((job_idx+1))/${total}] GPU=${gpu} tau=${tau} beta=${beta}"
    bash ./scripts/test_ceo.sh I ${gpu} ${P} ${tau} ${beta} 100.0 test ${MAX_SAMPLES} ${BS} \
      > ${LOG_DIR}/I_p${P}_tau${tau}_beta${beta}.txt 2>&1 &
    job_idx=$((job_idx + 1))
    if (( job_idx % NUM_GPUS == 0 )); then
      wait
    fi
  done
done
wait

echo ""
echo "=== OOD grid search done (${total} combos) ==="
echo "Next: python select_best_params.py --mode ood"
