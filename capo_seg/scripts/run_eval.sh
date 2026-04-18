#!/bin/bash
# >>> capo/scripts/_paths.sh (auto-inserted)
# Locate scripts/_paths.sh relative to this script's location so the file
# stays portable after a git clone -- no absolute paths.
_CAPO_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
source "${_CAPO_SCRIPT_DIR}/../../scripts/_paths.sh"
# <<< capo/scripts/_paths.sh
cd "$(dirname "$0")"

mkdir -p logs

TORCH_HOME=${TORCH_HOME}

declare -A CONFIGS
CONFIGS["r50_ade20k"]="configs/maskclip/maskclip_r50_512x512_ade20k.py|pretrain/RN50_clip_backbone.pth"
CONFIGS["vit16_ade20k"]="configs/maskclip/maskclip_vit16_512x512_ade20k.py|pretrain/ViT16_clip_backbone.pth"
CONFIGS["r50_voc12aug"]="configs/maskclip/maskclip_r50_512x512_voc12aug_20.py|pretrain/RN50_clip_backbone.pth"
CONFIGS["vit16_voc12aug"]="configs/maskclip/maskclip_vit16_512x512_voc12aug_20.py|pretrain/ViT16_clip_backbone.pth"

GPU=0
PIDS=()

for NAME in r50_ade20k vit16_ade20k r50_voc12aug vit16_voc12aug; do
    IFS='|' read -r CFG CKPT <<< "${CONFIGS[$NAME]}"
    LOG="logs/maskclip_${NAME}.log"

    echo "[GPU ${GPU}] ${NAME} -> ${LOG}"
    CUDA_VISIBLE_DEVICES=${GPU} TORCH_HOME=${TORCH_HOME} \
        python tools/test.py "${CFG}" "${CKPT}" --eval mIoU \
        > "${LOG}" 2>&1 &
    PIDS+=($!)
    GPU=$((GPU + 1))
done

echo ""
echo "All 4 jobs launched (PIDs: ${PIDS[*]})"
echo "Waiting for all jobs to finish..."

FAIL=0
for i in "${!PIDS[@]}"; do
    wait "${PIDS[$i]}"
    CODE=$?
    NAMES=(r50_ade20k vit16_ade20k r50_voc12aug vit16_voc12aug)
    if [ ${CODE} -eq 0 ]; then
        echo "[DONE] ${NAMES[$i]} (exit ${CODE})"
    else
        echo "[FAIL] ${NAMES[$i]} (exit ${CODE})"
        FAIL=$((FAIL + 1))
    fi
done

echo ""
if [ ${FAIL} -eq 0 ]; then
    echo "=== All 4 evaluations completed successfully ==="
else
    echo "=== ${FAIL} job(s) failed, check logs/ ==="
fi

echo ""
echo "=== Results Summary ==="
for NAME in r50_ade20k vit16_ade20k r50_voc12aug vit16_voc12aug; do
    LOG="logs/maskclip_${NAME}.log"
    echo "--- ${NAME} ---"
    grep -E "mIoU|aAcc|mAcc" "${LOG}" 2>/dev/null | tail -3
    echo ""
done
