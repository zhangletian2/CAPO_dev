#!/bin/bash
# >>> capo/scripts/_paths.sh (auto-inserted)
# Locate scripts/_paths.sh relative to this script's location so the file
# stays portable after a git clone -- no absolute paths.
_CAPO_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
source "${_CAPO_SCRIPT_DIR}/../../scripts/_paths.sh"
# <<< capo/scripts/_paths.sh
# Multi-task PoC: p_sh + Δ_t  (KD + L2 reg)
# Phase 1: DTD 20-shot/class + COCO val subset → p_sh, Δ_t
# Phase 2: DTD test + COCO 5K test (1:1 interleaved, bs=1)
#          Cloud uses dtd/new & coco2014_cloud image roots
#          3 configs run in parallel on 3 GPUs
#
# Usage:
#   bash run_multi_task_poc.sh              # 5K full (3-GPU parallel)
#   bash run_multi_task_poc.sh quick        # quick smoke test (single GPU)

set -e
MODE=${1:-full}
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

P1_ALPHA=1.0
P1_LAMBDA=0.01
P1_KD_TAU=0.4

run_one() {
    local GPU=$1 CONFIG=$2 CLS_N=$3 RET_N=$4 COCO_ANNO=$5
    local P1_EPOCHS=$6 P1_COCO_N=$7 P1_DTD_SHOT=$8 P1_CKPT=$9

    python multi_task_poc.py \
        --gpu "${GPU}" \
        --cloud_arch "ViT-B/16" --edge_arch "RN50" \
        --n_ctx 4 --ctx_init "a photo of a" \
        --lr 5e-3 --tta_steps 1 --tau 0.4 --beta 0.9 \
        --ent_top_k 20 --sim_top_k 256 \
        --cls_n "${CLS_N}" --ret_n "${RET_N}" --seed 42 \
        --coco_anno "${COCO_ANNO}" \
        --phase1_epochs "${P1_EPOCHS}" --phase1_coco_n "${P1_COCO_N}" \
        --phase1_dtd_shot "${P1_DTD_SHOT}" \
        --phase1_alpha ${P1_ALPHA} --phase1_lambda ${P1_LAMBDA} \
        --phase1_kd_tau ${P1_KD_TAU} \
        --phase1_ckpt "${P1_CKPT}" \
        --config "${CONFIG}"
}

if [ "$MODE" = "quick" ]; then
    # ── Quick smoke test (single GPU) ──
    CKPT="checkpoints/p1_quick.pt"
    ANNO="${RET_ROOT}/dataset/coco2014/coco_karpathy_test.json"
    mkdir -p checkpoints

    echo "════════════════════════════════════════════════════"
    echo " Quick smoke test  GPU=0"
    echo " DTD 200  COCO 100  Phase1 3ep 5-shot"
    echo "════════════════════════════════════════════════════"

    run_one 0 task_init 200 100 "$ANNO" 3 200 5 "$CKPT"
    echo "Done."
    exit 0
fi

# ═══════════════════════════════════════════════════════════
# Full 5K evaluation — 3-GPU parallel
# ═══════════════════════════════════════════════════════════
GPU0=3; GPU1=4; GPU2=5
CLS_N=1692; RET_N=5000
COCO_ANNO="${RET_ROOT}/dataset/coco2014/coco_karpathy_test.json"
P1_EPOCHS=3; P1_COCO_N=200; P1_DTD_SHOT=5
P1_CKPT="checkpoints/p1_s${P1_DTD_SHOT}_c${P1_COCO_N}_ep${P1_EPOCHS}.pt"

mkdir -p checkpoints logs

echo "════════════════════════════════════════════════════"
echo " Multi-Task PoC 5K — 3-GPU parallel"
echo " Phase1: ${P1_DTD_SHOT}-shot DTD + COCO ${P1_COCO_N} × ${P1_EPOCHS}ep"
echo " Phase2: DTD ${CLS_N} + COCO 5K (3 configs)"
echo " Checkpoint: ${P1_CKPT}"
echo "════════════════════════════════════════════════════"

# ── Step 1: Phase 1 on GPU 0 → also runs task_init ──
echo "[$(date +%H:%M:%S)] Phase 1 + task_init on GPU ${GPU0} ..."
run_one ${GPU0} task_init ${CLS_N} ${RET_N} "${COCO_ANNO}" \
    ${P1_EPOCHS} ${P1_COCO_N} ${P1_DTD_SHOT} "${P1_CKPT}" \
    2>&1 | tee logs/task_init.log &
PID0=$!

# Wait for Phase 1 checkpoint
while [ ! -f "${P1_CKPT}" ]; do sleep 2; done
sleep 2
echo "[$(date +%H:%M:%S)] Phase 1 checkpoint ready."

# ── Step 2: remaining 2 configs on GPU 1 & 2 ──
echo "[$(date +%H:%M:%S)] shared_init  on GPU ${GPU1} ..."
run_one ${GPU1} shared_init ${CLS_N} ${RET_N} "${COCO_ANNO}" \
    ${P1_EPOCHS} ${P1_COCO_N} ${P1_DTD_SHOT} "${P1_CKPT}" \
    2>&1 | tee logs/shared_init.log &
PID1=$!

echo "[$(date +%H:%M:%S)] task_residual on GPU ${GPU2} ..."
run_one ${GPU2} task_residual ${CLS_N} ${RET_N} "${COCO_ANNO}" \
    ${P1_EPOCHS} ${P1_COCO_N} ${P1_DTD_SHOT} "${P1_CKPT}" \
    2>&1 | tee logs/task_residual.log &
PID2=$!

echo "[$(date +%H:%M:%S)] All 3 configs running. Waiting ..."

wait $PID0 && echo "[$(date +%H:%M:%S)] task_init     ✓ (GPU ${GPU0})"
wait $PID1 && echo "[$(date +%H:%M:%S)] shared_init   ✓ (GPU ${GPU1})"
wait $PID2 && echo "[$(date +%H:%M:%S)] task_residual ✓ (GPU ${GPU2})"

echo ""
echo "════════════════════════════════════════════════════"
echo " Done.  Logs → logs/  Results → multi_task_poc_result_*.json"
echo "════════════════════════════════════════════════════"
