#!/bin/bash
# >>> capo/scripts/_paths.sh (auto-inserted)
# Locate scripts/_paths.sh relative to this script's location so the file
# stays portable after a git clone -- no absolute paths.
_CAPO_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
source "${_CAPO_SCRIPT_DIR}/../../scripts/_paths.sh"
# <<< capo/scripts/_paths.sh
# OOD evaluation with new_qp7 reconstructed images (4 datasets, no K).
# 3 GPUs: I alone, A→V serial, R alone.
#
# Usage: bash run_final_ood_qp7.sh <gpu_I> <gpu_AV> <gpu_R> <selection_p> <tau> <beta>
# Example: bash run_final_ood_qp7.sh 0 1 2 0.1 0.2 0.9

GPU_I=${1:?  "Usage: bash run_final_ood_qp7.sh <gpu_I> <gpu_AV> <gpu_R> <p> <tau> <beta>"}
GPU_AV=${2:? "Missing gpu for A/V"}
GPU_R=${3:?  "Missing gpu for R"}
P=${4:?      "Missing selection_p"}
TAU=${5:?    "Missing tau"}
BETA=${6:?   "Missing beta"}

REC_DIR=new_qp7
LOG_DIR=./logs/final_ood_qp7
mkdir -p ${LOG_DIR}

run_ds() {
    local ds=$1 gpu=$2
    bash ./scripts/test_ceo.sh ${ds} ${gpu} ${P} ${TAU} ${BETA} 100.0 test 0 64 0 ${REC_DIR} \
        > ${LOG_DIR}/${ds}_p${P}_tau${TAU}_beta${BETA}.txt 2>&1
}

echo "=== Final OOD Evaluation (rec_dir=${REC_DIR}, no K) ==="
echo "Params: p=${P}, tau=${TAU}, beta=${BETA}"
echo "GPU ${GPU_I}: I | GPU ${GPU_AV}: A→V | GPU ${GPU_R}: R"
echo ""

# GPU0: I (alone, 50k images)
echo "[GPU ${GPU_I}] Evaluating I ..."
run_ds I ${GPU_I} &

# GPU1: A → V (serial, 7.5k + 10k)
(
    echo "[GPU ${GPU_AV}] Evaluating A ..."
    run_ds A ${GPU_AV}
    echo "[GPU ${GPU_AV}] A done → V ..."
    run_ds V ${GPU_AV}
    echo "[GPU ${GPU_AV}] V done"
) &

# GPU2: R (alone, 30k images)
echo "[GPU ${GPU_R}] Evaluating R ..."
run_ds R ${GPU_R} &

wait

echo ""
echo "=== OOD (${REC_DIR}) Evaluation Done ==="
for ds in I A R V; do
    echo -n "${ds}: "
    tail -1 ${LOG_DIR}/${ds}_p${P}_tau${TAU}_beta${BETA}.txt
done
echo "Results in ${LOG_DIR}/"
