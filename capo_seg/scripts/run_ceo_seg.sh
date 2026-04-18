#!/bin/bash
# >>> capo/scripts/_paths.sh (auto-inserted)
# Locate scripts/_paths.sh relative to this script's location so the file
# stays portable after a git clone -- no absolute paths.
_CAPO_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
source "${_CAPO_SCRIPT_DIR}/../../scripts/_paths.sh"
# <<< capo/scripts/_paths.sh
set -euo pipefail
cd "$(dirname "$0")"

mkdir -p logs

PYTHON=${PYTHON:-python}
CLIP_ROOT=${CLIP_CACHE}
export TORCH_HOME=${TORCH_HOME}

# ---- Defaults ----
TTA_STEPS=1
LR=5e-3
SEL_P=0.1
TAU=0.4
BETA=0.9
N_CTX=4

usage() {
    echo "Usage: $0 [OPTIONS]"
    echo "  --tta_steps  N   TTA optimisation steps per image (default: 1)"
    echo "  --lr         F   Prompt learning rate              (default: 5e-3)"
    echo "  --sel_p      F   Confident pixel fraction          (default: 0.1)"
    echo "  --tau        F   KL temperature                    (default: 0.4)"
    echo "  --beta       F   KL loss weight                    (default: 0.9)"
    echo "  --n_ctx      N   Learnable prompt tokens           (default: 4)"
    exit 0
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --tta_steps) TTA_STEPS="$2"; shift 2;;
        --lr)        LR="$2";        shift 2;;
        --sel_p)     SEL_P="$2";     shift 2;;
        --tau)       TAU="$2";       shift 2;;
        --beta)      BETA="$2";      shift 2;;
        --n_ctx)     N_CTX="$2";     shift 2;;
        -h|--help)   usage;;
        *) echo "Unknown option: $1"; usage;;
    esac
done

TAG="steps${TTA_STEPS}_lr${LR}_p${SEL_P}_tau${TAU}_beta${BETA}_ctx${N_CTX}"

echo "=============================="
echo "CEO-Seg TTA Evaluation"
echo "  tta_steps=${TTA_STEPS}  lr=${LR}  selection_p=${SEL_P}"
echo "  tau=${TAU}  beta=${BETA}  n_ctx=${N_CTX}"
echo "  tag=${TAG}"
echo "=============================="

# ---- ADE20K (GPU 0) ----
echo "[GPU 0] ADE20K -> logs/ceo_seg_ade20k_${TAG}.log"
CUDA_VISIBLE_DEVICES=0 ${PYTHON} -u tools/ceo_seg.py \
    --cloud-config configs/maskclip/maskclip_vit16_512x512_ade20k.py \
    --cloud-ckpt  pretrain/ViT16_clip_backbone.pth \
    --edge-config configs/maskclip/maskclip_r50_512x512_ade20k.py \
    --edge-ckpt   pretrain/RN50_clip_backbone.pth \
    --cloud-arch ViT-B/16 --edge-arch RN50 --dataset ade \
    --tta_steps ${TTA_STEPS} --lr ${LR} --selection_p ${SEL_P} \
    --tau ${TAU} --beta ${BETA} --n_ctx ${N_CTX} \
    --clip-download-root ${CLIP_ROOT} --gpu 0 \
    > "logs/ceo_seg_ade20k_${TAG}.log" 2>&1 &
PID_ADE=$!

# ---- VOC (GPU 1) ----
echo "[GPU 1] VOC   -> logs/ceo_seg_voc_${TAG}.log"
CUDA_VISIBLE_DEVICES=1 ${PYTHON} -u tools/ceo_seg.py \
    --cloud-config configs/maskclip/maskclip_vit16_512x512_voc12aug_20.py \
    --cloud-ckpt  pretrain/ViT16_clip_backbone.pth \
    --edge-config configs/maskclip/maskclip_r50_512x512_voc12aug_20.py \
    --edge-ckpt   pretrain/RN50_clip_backbone.pth \
    --cloud-arch ViT-B/16 --edge-arch RN50 --dataset voc \
    --tta_steps ${TTA_STEPS} --lr ${LR} --selection_p ${SEL_P} \
    --tau ${TAU} --beta ${BETA} --n_ctx ${N_CTX} \
    --clip-download-root ${CLIP_ROOT} --gpu 0 \
    > "logs/ceo_seg_voc_${TAG}.log" 2>&1 &
PID_VOC=$!

echo ""
echo "Both jobs launched (ADE PID=${PID_ADE}, VOC PID=${PID_VOC})"
echo "Waiting..."

FAIL=0
wait ${PID_ADE}; C=$?; [ $C -ne 0 ] && FAIL=$((FAIL+1)) && echo "[FAIL] ADE20K (exit $C)" || echo "[DONE] ADE20K"
wait ${PID_VOC}; C=$?; [ $C -ne 0 ] && FAIL=$((FAIL+1)) && echo "[FAIL] VOC    (exit $C)" || echo "[DONE] VOC"

echo ""
echo "========== Results =========="
for DS in ade20k voc; do
    LOG="logs/ceo_seg_${DS}_${TAG}.log"
    echo "--- ${DS} ---"
    grep -E "mIoU|mAcc|aAcc" "${LOG}" 2>/dev/null | tail -5
    echo ""
done

[ ${FAIL} -eq 0 ] && echo "All done." || echo "${FAIL} job(s) failed, check logs/"
