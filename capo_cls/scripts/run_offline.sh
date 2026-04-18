#!/bin/bash
# >>> capo/scripts/_paths.sh (auto-inserted)
# Locate scripts/_paths.sh relative to this script's location so the file
# stays portable after a git clone -- no absolute paths.
_CAPO_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
source "${_CAPO_SCRIPT_DIR}/../../scripts/_paths.sh"
# <<< capo/scripts/_paths.sh
# CEO Offline: train on ImageNet val (few-shot), eval on 15 test sets.
#
# Supports multi-GPU parallel — each shot gets its own GPU.
#
# Usage:
#   bash scripts/run_offline.sh 2              # 1 GPU, 5 shots sequential
#   bash scripts/run_offline.sh 2,3            # 2 GPUs, round-robin
#   bash scripts/run_offline.sh 2,3,4,5,6      # 5 GPUs, fully parallel
#   bash scripts/run_offline.sh 2,3 new_qp7    # custom rec_dir
#
# Environment: conda activate ceo

set -euo pipefail

GPUS=${1:-0}
REC_DIR=${2:-new}
EPOCHS=${3:-5}
LR=${4:-2e-3}
TAU=0.4
BETA=0.9
SEED=0

IFS=',' read -ra GPU_LIST <<< "${GPUS}"
N_GPU=${#GPU_LIST[@]}

LOG_DIR=./logs/offline
mkdir -p ${LOG_DIR}

SHOTS=(1 2 4 8 16)
PIDS=()

echo "═══════════════════════════════════════════════════"
echo "  CEO Offline — ${#SHOTS[@]} shots × ${N_GPU} GPU(s)"
echo "  GPUs: ${GPUS}   rec_dir: ${REC_DIR}"
echo "  epochs: ${EPOCHS}  lr: ${LR}  tau: ${TAU}  beta: ${BETA}"
echo "═══════════════════════════════════════════════════"

for i in "${!SHOTS[@]}"; do
    SHOT=${SHOTS[$i]}
    GPU=${GPU_LIST[$(( i % N_GPU ))]}
    TAG="shot${SHOT}_ep${EPOCHS}_tau${TAU}_beta${BETA}_lr${LR}_seed${SEED}_rec${REC_DIR}"
    LOG="${LOG_DIR}/${TAG}.txt"

    echo "[shot=${SHOT}]  GPU ${GPU}  →  ${LOG}"
    python ceo_offline.py ./dataset \
        --n_shot ${SHOT} \
        --epochs ${EPOCHS} \
        --lr ${LR} \
        --tau ${TAU} \
        --beta ${BETA} \
        --gpu ${GPU} \
        --seed ${SEED} \
        --rec_dir ${REC_DIR} \
        > "${LOG}" 2>&1 &
    PIDS+=($!)
done

echo ""
echo "All ${#SHOTS[@]} jobs launched.  Waiting …"

FAIL=0
for i in "${!PIDS[@]}"; do
    PID=${PIDS[$i]}
    SHOT=${SHOTS[$i]}
    if wait ${PID}; then
        echo "  ✓ shot=${SHOT} (pid ${PID}) finished"
    else
        echo "  ✗ shot=${SHOT} (pid ${PID}) FAILED (exit $?)"
        FAIL=1
    fi
done

echo ""
echo "═══ Summary ═══"
echo ""
printf "%-6s" "shot"
printf "%-10s %-10s" "I_top1" "A_top1"
printf "%-10s %-10s" "R_top1" "V_top1"
printf "%-10s" "K_top1"
echo ""
echo "────────────────────────────────────────────────────────"

for SHOT in "${SHOTS[@]}"; do
    CKPT_DIR=$(python -c "
lr=float('${LR}'); tau=float('${TAU}'); beta=float('${BETA}')
print(f'checkpoints/offline_${SHOT}shot_ep${EPOCHS}_tau{tau}_beta{beta}_lr{lr}_seed${SEED}')
" 2>/dev/null)
    RESULT="${CKPT_DIR}/results.json"
    if [ -f "${RESULT}" ]; then
        printf "%-6s" "${SHOT}"
        for DS in I A R V K; do
            VAL=$(python -c "import json; d=json.load(open('${RESULT}')); print(f\"{d['${DS}']['top1']:.2f}\")" 2>/dev/null || echo "N/A")
            printf "%-10s" "${VAL}"
        done
        echo ""
    else
        printf "%-6s  (no results)\n" "${SHOT}"
    fi
done

echo ""
echo "Logs:    ${LOG_DIR}/"
echo "Prompts: checkpoints/offline_*shot_*/"

if [ ${FAIL} -ne 0 ]; then
    echo "WARNING: some jobs failed — check logs above."
    exit 1
fi
