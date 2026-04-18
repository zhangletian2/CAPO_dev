#!/bin/bash
# >>> capo/scripts/_paths.sh (auto-inserted)
# Locate scripts/_paths.sh relative to this script's location so the file
# stays portable after a git clone -- no absolute paths.
_CAPO_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
source "${_CAPO_SCRIPT_DIR}/../../scripts/_paths.sh"
# <<< capo/scripts/_paths.sh
# Ablation: cloud model uses original images (same as edge) instead of reconstructed.
# 3 GPUs: I alone, K alone, A→R→V serial on the third.
#
# Usage: bash run_final_ood_ori.sh <gpu0> <gpu1> <gpu2> <selection_p> <tau> <beta>
# Example: bash run_final_ood_ori.sh 0 1 2 0.1 0.2 0.9

GPU_I=${1:?  "Usage: bash run_final_ood_ori.sh <gpu_I> <gpu_K> <gpu_ARV> <p> <tau> <beta>"}
GPU_K=${2:?  "Missing gpu for K"}
GPU_ARV=${3:? "Missing gpu for A/R/V"}
P=${4:?      "Missing selection_p"}
TAU=${5:?    "Missing tau"}
BETA=${6:?   "Missing beta"}

LOG_DIR=./logs/final_ood_ori
mkdir -p ${LOG_DIR}

run_ds() {
    local ds=$1 gpu=$2
    bash ./scripts/test_ceo.sh ${ds} ${gpu} ${P} ${TAU} ${BETA} 100.0 test 0 64 1 \
        > ${LOG_DIR}/${ds}_p${P}_tau${TAU}_beta${BETA}.txt 2>&1
}

echo "=== Final OOD Evaluation (cloud_use_ori) ==="
echo "Params: p=${P}, tau=${TAU}, beta=${BETA}"
echo "GPU ${GPU_I}: I | GPU ${GPU_K}: K | GPU ${GPU_ARV}: A→R→V"
echo ""

# GPU0: I (alone)
echo "[GPU ${GPU_I}] Evaluating I ..."
run_ds I ${GPU_I} &

# GPU1: K (alone)
echo "[GPU ${GPU_K}] Evaluating K ..."
run_ds K ${GPU_K} &

# GPU2: A → R → V (serial)
(
    echo "[GPU ${GPU_ARV}] Evaluating A ..."
    run_ds A ${GPU_ARV}
    echo "[GPU ${GPU_ARV}] A done → R ..."
    run_ds R ${GPU_ARV}
    echo "[GPU ${GPU_ARV}] R done → V ..."
    run_ds V ${GPU_ARV}
    echo "[GPU ${GPU_ARV}] V done"
) &

wait

echo ""
echo "=== OOD (cloud_use_ori) Evaluation Done ==="
for ds in I A R K V; do
    echo -n "${ds}: "
    tail -1 ${LOG_DIR}/${ds}_p${P}_tau${TAU}_beta${BETA}.txt
done
echo "Results in ${LOG_DIR}/"
