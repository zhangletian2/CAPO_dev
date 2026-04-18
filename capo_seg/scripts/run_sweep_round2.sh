#!/bin/bash
# >>> capo/scripts/_paths.sh (auto-inserted)
# Locate scripts/_paths.sh relative to this script's location so the file
# stays portable after a git clone -- no absolute paths.
_CAPO_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
source "${_CAPO_SCRIPT_DIR}/../../scripts/_paths.sh"
# <<< capo/scripts/_paths.sh
# Round 2: Sweep tau (3) + selection_p (3), fix beta & cloud_conf_p to Round 1 best
# >>> BEFORE RUNNING: Update BEST_BETA and BEST_CP from Round 1 results <<<
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs

PY=${PY:-python}
export TORCH_HOME=${TORCH_HOME}
CLIP_ROOT=${CLIP_CACHE}

# ====== Fill in from Round 1 best ======
BEST_BETA=0.9
BEST_CP=0.1
# ========================================

COMMON="--cloud-config configs/maskclip/maskclip_vit16_512x512_voc12aug_20_npe.py \
--cloud-ckpt pretrain/ViT16_clip_backbone.pth \
--edge-config configs/maskclip/maskclip_r50_512x512_voc12aug_20_npe.py \
--edge-ckpt pretrain/RN50_clip_backbone.pth \
--cloud-arch ViT-B/16 --edge-arch RN50 --dataset voc \
--ctx_init a_photo_of_a --clip-download-root $CLIP_ROOT --gpu 0 \
--tta_steps 1 --selective_kl --beta $BEST_BETA --cloud_conf_p $BEST_CP"

run() {
    local GPU=$1 TAG=$2; shift 2
    local LOG="logs/sweep2_${TAG}.log"
    echo "[GPU $GPU] $TAG -> $LOG"
    CUDA_VISIBLE_DEVICES=$GPU $PY -u tools/ceo_seg.py $COMMON "$@" > "$LOG" 2>&1 &
    eval "PID_${GPU}=$!"
}

echo "=========================================="
echo "Sweep Round 2: tau x selection_p"
echo "fix: beta=$BEST_BETA, cloud_conf_p=$BEST_CP"
echo "=========================================="

run 0 "t0.1_sp0.05"   --tau 0.1 --selection_p 0.05
run 1 "t0.1_sp0.1"    --tau 0.1 --selection_p 0.1
run 2 "t0.1_sp0.2"    --tau 0.1 --selection_p 0.2
run 3 "t0.4_sp0.05"   --tau 0.4 --selection_p 0.05
# t0.4_sp0.1 = Round 1 best (already have)
run 4 "t0.4_sp0.2"    --tau 0.4 --selection_p 0.2
run 5 "t1.0_sp0.1"    --tau 1.0 --selection_p 0.1

echo ""
echo "All 6 jobs launched. Waiting..."

FAIL=0
for G in 0 1 2 3 4 5; do
    eval "PID=\$PID_${G}"
    wait $PID; C=$?
    [ $C -ne 0 ] && FAIL=$((FAIL+1)) && echo "[FAIL] GPU $G (exit $C)" || echo "[DONE] GPU $G"
done

echo ""
echo "========== Sweep Round 2 Results =========="
echo "fix: beta=$BEST_BETA, cloud_conf_p=$BEST_CP"
echo "Reference: ent_only=46.10 | Round1_best=?? | ViT16=47.57"
echo ""
printf "%-20s %s\n" "Experiment" "Result"
printf "%-20s %s\n" "--------------------" "----------"
for F in t0.1_sp0.05 t0.1_sp0.1 t0.1_sp0.2 t0.4_sp0.05 t0.4_sp0.2 t1.0_sp0.1; do
    LOG="logs/sweep2_${F}.log"
    printf "%-20s " "$F"
    grep -oP 'mIoU.*?(\d+\.\d+)' "$LOG" 2>/dev/null | head -1 || echo "(no result)"
done
echo "t0.4_sp0.1 (prev)    = Round 1 best"

echo ""
[ $FAIL -eq 0 ] && echo "All done." || echo "$FAIL job(s) failed."
