#!/bin/bash
# >>> capo/scripts/_paths.sh (auto-inserted)
# Locate scripts/_paths.sh relative to this script's location so the file
# stays portable after a git clone -- no absolute paths.
_CAPO_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
source "${_CAPO_SCRIPT_DIR}/../../scripts/_paths.sh"
# <<< capo/scripts/_paths.sh
# Ablation: cloud model uses original images (same as edge) instead of reconstructed.
# 4 param combos × 10 datasets on 4 GPUs, event-driven parallel scheduling.
#
# Schedule per GPU (same as run_final_fewshot.sh):
#   t=0:    food101 + sun397 + eurosat + ucf101
#   ucf101 done     → pets
#   pets done       → aircraft
#   eurosat done    → dtd
#   aircraft done   → cars
#   cars done       → caltech101
#   sun397 done     → flower102
#   food101 done    → all done
#
# Usage: bash run_final_fewshot_ori.sh

P=0.1
LOG_DIR=./logs/final_fewshot_ori
mkdir -p ${LOG_DIR}

declare -a GPUS=(3)
declare -a TAUS=(0.5)
declare -a BETAS=(0.8)

run_ds() {
    local ds=$1 gpu=$2 tau=$3 beta=$4
    bash ./scripts/test_ceo.sh ${ds} ${gpu} ${P} ${tau} ${beta} 100.0 test 0 64 1 \
        > ${LOG_DIR}/${ds}_p${P}_tau${tau}_beta${beta}.txt 2>&1
}

schedule_one_gpu() {
    local gpu=$1 tau=$2 beta=$3

    echo "[GPU ${gpu}] tau=${tau}, beta=${beta} — start (cloud_use_ori)"

    # Wave 1: food101 + sun397 + eurosat + ucf101
    run_ds food101  ${gpu} ${tau} ${beta} &
    local PF=$!
    run_ds sun397   ${gpu} ${tau} ${beta} &
    local PS=$!
    run_ds eurosat  ${gpu} ${tau} ${beta} &
    local PE=$!
    run_ds ucf101   ${gpu} ${tau} ${beta} &
    local PU=$!

    # ucf101 done → pets
    wait $PU
    echo "[GPU ${gpu}] ucf101 done → pets"
    run_ds pets ${gpu} ${tau} ${beta} &
    local PP=$!

    # pets done → aircraft
    wait $PP
    echo "[GPU ${gpu}] pets done → aircraft"
    run_ds aircraft ${gpu} ${tau} ${beta} &
    local PA=$!

    # eurosat done → dtd
    wait $PE
    echo "[GPU ${gpu}] eurosat done → dtd"
    run_ds dtd ${gpu} ${tau} ${beta} &
    local PD=$!

    # aircraft done → cars
    wait $PA
    echo "[GPU ${gpu}] aircraft done → cars"
    run_ds cars ${gpu} ${tau} ${beta} &
    local PC=$!

    # cars done → caltech101
    wait $PC
    echo "[GPU ${gpu}] cars done → caltech101"
    run_ds caltech101 ${gpu} ${tau} ${beta} &
    local PK=$!

    # sun397 done → flower102
    wait $PS
    echo "[GPU ${gpu}] sun397 done → flower102"
    run_ds flower102 ${gpu} ${tau} ${beta} &
    local PO=$!

    # wait remaining: food101, dtd, caltech101, flower102
    wait $PF $PD $PK $PO
    echo "[GPU ${gpu}] tau=${tau}, beta=${beta} — ALL DONE"
}

echo "=== Final Fine-grained Eval (cloud_use_ori): 4 combos × 10 datasets ==="
echo "GPUs: ${GPUS[*]}"
echo "Combos: (tau,beta) = (${TAUS[0]},${BETAS[0]}) (${TAUS[1]},${BETAS[1]}) (${TAUS[2]},${BETAS[2]}) (${TAUS[3]},${BETAS[3]})"
echo ""

for i in 0; do
    schedule_one_gpu ${GPUS[$i]} ${TAUS[$i]} ${BETAS[$i]} &
done

wait

echo ""
echo "=== All done (cloud_use_ori) ==="
echo "Results: ${LOG_DIR}/"
