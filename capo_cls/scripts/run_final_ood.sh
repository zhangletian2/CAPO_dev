#!/bin/bash
# >>> capo/scripts/_paths.sh (auto-inserted)
# Locate scripts/_paths.sh relative to this script's location so the file
# stays portable after a git clone -- no absolute paths.
_CAPO_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
source "${_CAPO_SCRIPT_DIR}/../../scripts/_paths.sh"
# <<< capo/scripts/_paths.sh
# Final OOD evaluation with best params selected from ImageNet val.
# Supports multi-GPU parallel execution.
#
# Usage: bash run_final_ood.sh <gpu_list> <selection_p> <tau> <beta>
# Example: bash run_final_ood.sh "0,1,2,3,4" 0.1 0.6 0.9

GPU_LIST=${1:? "Usage: bash run_final_ood.sh <gpu_list> <p> <tau> <beta>"}
P=${2:?       "Missing selection_p"}
TAU=${3:?     "Missing tau"}
BETA=${4:?    "Missing beta"}

IFS=',' read -ra GPUS <<< "${GPU_LIST}"
NUM_GPUS=${#GPUS[@]}

LOG_DIR=./logs/final_ood
mkdir -p ${LOG_DIR}

DATASETS=(I A R K V)

echo "=== Final OOD Evaluation ==="
echo "Params: p=${P}, tau=${TAU}, beta=${BETA}, GPUs: ${GPU_LIST}"
echo ""

job_idx=0
for ds in "${DATASETS[@]}"; do
    gpu=${GPUS[$((job_idx % NUM_GPUS))]}
    echo "[GPU ${gpu}] Evaluating ${ds} (test) ..."
    bash ./scripts/test_ceo.sh ${ds} ${gpu} ${P} ${TAU} ${BETA} 100.0 test 0 64 \
        > ${LOG_DIR}/${ds}_p${P}_tau${TAU}_beta${BETA}.txt 2>&1 &
    job_idx=$((job_idx + 1))
    if (( job_idx % NUM_GPUS == 0 )); then
        wait
    fi
done
wait

echo ""
echo "=== OOD Final Evaluation Done ==="
for ds in "${DATASETS[@]}"; do
    echo -n "${ds}: "
    tail -1 ${LOG_DIR}/${ds}_p${P}_tau${TAU}_beta${BETA}.txt
done
echo "Results in ${LOG_DIR}/"
