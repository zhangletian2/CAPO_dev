#!/bin/bash
# >>> capo/scripts/_paths.sh (auto-inserted)
# Locate scripts/_paths.sh relative to this script's location so the file
# stays portable after a git clone -- no absolute paths.
_CAPO_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
source "${_CAPO_SCRIPT_DIR}/../../scripts/_paths.sh"
# <<< capo/scripts/_paths.sh
# Round 1: Sweep beta (3) + cloud_conf_p (3), fix tau=0.4, selection_p=0.1
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs

PY=${PY:-python}
export TORCH_HOME=${TORCH_HOME}
CLIP_ROOT=${CLIP_CACHE}

COMMON="--cloud-config configs/maskclip/maskclip_vit16_512x512_voc12aug_20_npe.py \
--cloud-ckpt pretrain/ViT16_clip_backbone.pth \
--edge-config configs/maskclip/maskclip_r50_512x512_voc12aug_20_npe.py \
--edge-ckpt pretrain/RN50_clip_backbone.pth \
--cloud-arch ViT-B/16 --edge-arch RN50 --dataset voc \
--ctx_init a_photo_of_a --clip-download-root $CLIP_ROOT --gpu 0 \
--tta_steps 1 --selective_kl --tau 0.4 --selection_p 0.1"

run() {
    local GPU=$1 TAG=$2; shift 2
    local LOG="logs/sweep1_${TAG}.log"
    echo "[GPU $GPU] $TAG -> $LOG"
    CUDA_VISIBLE_DEVICES=$GPU $PY -u tools/ceo_seg.py $COMMON "$@" > "$LOG" 2>&1 &
    eval "PID_${GPU}=$!"
}

echo "=========================================="
echo "Sweep Round 1: beta x cloud_conf_p"
echo "fix: tau=0.4, selection_p=0.1"
echo "=========================================="

run 0 "b0.5_cp0.1"   --beta 0.5 --cloud_conf_p 0.1
run 1 "b0.5_cp0.3"   --beta 0.5 --cloud_conf_p 0.3
run 2 "b0.5_cp0.5"   --beta 0.5 --cloud_conf_p 0.5
run 3 "b0.9_cp0.1"   --beta 0.9 --cloud_conf_p 0.1
# b0.9_cp0.3 = 47.50 (already have)
run 4 "b0.9_cp0.5"   --beta 0.9 --cloud_conf_p 0.5
run 5 "b2.0_cp0.3"   --beta 2.0 --cloud_conf_p 0.3

echo ""
echo "All 6 jobs launched. Waiting..."

FAIL=0
for G in 0 1 2 3 4 5; do
    eval "PID=\$PID_${G}"
    wait $PID; C=$?
    [ $C -ne 0 ] && FAIL=$((FAIL+1)) && echo "[FAIL] GPU $G (exit $C)" || echo "[DONE] GPU $G"
done

echo ""
echo "========== Sweep Round 1 Results =========="
echo "Reference: ent_only=46.10 | b0.9_cp0.3=47.50 | ViT16=47.57"
echo ""
printf "%-20s %s\n" "Experiment" "Result"
printf "%-20s %s\n" "--------------------" "----------"
for F in b0.5_cp0.1 b0.5_cp0.3 b0.5_cp0.5 b0.9_cp0.1 b0.9_cp0.5 b2.0_cp0.3; do
    LOG="logs/sweep1_${F}.log"
    printf "%-20s " "$F"
    grep -oP 'mIoU.*?(\d+\.\d+)' "$LOG" 2>/dev/null | head -1 || echo "(no result)"
done
echo "b0.9_cp0.3 (prev)    mIoU: 0.4750"

echo ""
[ $FAIL -eq 0 ] && echo "All done." || echo "$FAIL job(s) failed."
