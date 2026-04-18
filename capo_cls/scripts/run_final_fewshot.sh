#!/bin/bash
# >>> capo/scripts/_paths.sh (auto-inserted)
# Locate scripts/_paths.sh relative to this script's location so the file
# stays portable after a git clone -- no absolute paths.
_CAPO_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
source "${_CAPO_SCRIPT_DIR}/../../scripts/_paths.sh"
# <<< capo/scripts/_paths.sh
# Final fine-grained evaluation: 4 param combos × 10 datasets on 4 GPUs.
# Each GPU runs 1 combo with event-driven parallel scheduling (VRAM ≤ 20G).
#
# Schedule per GPU (greedy, peak 19.5G, bottleneck = food101 @ 505min ≈ 8.4h):
#   t=0:    food101(3.8) + sun397(9.1) + eurosat(2.2) + ucf101(3.8)  = 18.9G
#   ucf101 done     → pets(2.7)                                       18.9→17.8G
#   pets done       → aircraft(3.8)                                    17.8→18.9G
#   eurosat done    → dtd(2.8)                                         +dtd→19.5G
#   aircraft done   → cars(5.5)                                        swap→18.4G
#   cars done       → caltech101(3.8)                                  swap→16.7G
#   sun397 done     → flower102(3.8)                                   swap→11.4G
#   food101 done    → all done
#
# Usage: bash run_final_fewshot.sh

P=0.1
LOG_DIR=./logs/final_fewshot
mkdir -p ${LOG_DIR}

declare -a GPUS=(0 1)
declare -a TAUS=(0.4 0.4)
declare -a BETAS=(0.7 1.1)

run_ds() {
    local ds=$1 gpu=$2 tau=$3 beta=$4
    bash ./scripts/test_ceo.sh ${ds} ${gpu} ${P} ${tau} ${beta} 100.0 test 0 64 \
        > ${LOG_DIR}/${ds}_p${P}_tau${tau}_beta${beta}.txt 2>&1
}

schedule_one_gpu() {
    local gpu=$1 tau=$2 beta=$3

    echo "[GPU ${gpu}] tau=${tau}, beta=${beta} — start"

    # Wave 1: food101 + sun397 + eurosat + ucf101 (18.9G)
    run_ds food101  ${gpu} ${tau} ${beta} &
    local PF=$!
    run_ds sun397   ${gpu} ${tau} ${beta} &
    local PS=$!
    run_ds eurosat  ${gpu} ${tau} ${beta} &
    local PE=$!
    run_ds ucf101   ${gpu} ${tau} ${beta} &
    local PU=$!

    # ucf101 done (~63m) → pets (frees 3.8G, adds 2.7G → 17.8G)
    wait $PU
    echo "[GPU ${gpu}] ucf101 done → pets"
    run_ds pets ${gpu} ${tau} ${beta} &
    local PP=$!

    # pets done (~124m) → aircraft (frees 2.7G, adds 3.8G → 18.9G)
    wait $PP
    echo "[GPU ${gpu}] pets done → aircraft"
    run_ds aircraft ${gpu} ${tau} ${beta} &
    local PA=$!

    # eurosat done (~135m) → dtd (frees 2.2G, adds 2.8G → 19.5G)
    wait $PE
    echo "[GPU ${gpu}] eurosat done → dtd"
    run_ds dtd ${gpu} ${tau} ${beta} &
    local PD=$!

    # aircraft done (~180m) → cars (frees 3.8G, adds 5.5G → 18.4G)
    wait $PA
    echo "[GPU ${gpu}] aircraft done → cars"
    run_ds cars ${gpu} ${tau} ${beta} &
    local PC=$!

    # cars done (~314m) → caltech101 (frees 5.5G, adds 3.8G → 16.7G)
    wait $PC
    echo "[GPU ${gpu}] cars done → caltech101"
    run_ds caltech101 ${gpu} ${tau} ${beta} &
    local PK=$!

    # sun397 done (~331m) → flower102 (frees 9.1G, adds 3.8G → 11.4G)
    wait $PS
    echo "[GPU ${gpu}] sun397 done → flower102"
    run_ds flower102 ${gpu} ${tau} ${beta} &
    local PO=$!

    # wait remaining: food101, dtd, caltech101, flower102
    wait $PF $PD $PK $PO
    echo "[GPU ${gpu}] tau=${tau}, beta=${beta} — ALL DONE"
}

echo "=== Final Fine-grained Eval: 4 combos × 10 datasets ==="
echo "GPUs: ${GPUS[*]}"
echo "Combos: (tau,beta) = (${TAUS[0]},${BETAS[0]}) (${TAUS[1]},${BETAS[1]}) (${TAUS[2]},${BETAS[2]}) (${TAUS[3]},${BETAS[3]})"
echo "VRAM budget: 20G per GPU, peak usage: 19.5G"
echo "Estimated wall time: ~8.4h (bottleneck: food101 with 30300 images)"
echo ""

for i in 0 1; do
    schedule_one_gpu ${GPUS[$i]} ${TAUS[$i]} ${BETAS[$i]} &
done

wait

echo ""
echo "=== All done ==="
echo "Results: ${LOG_DIR}/"
echo "Next: python collect_final_results.py"
