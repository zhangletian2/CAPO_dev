#!/bin/bash
# >>> capo/scripts/_paths.sh (auto-inserted)
# Locate scripts/_paths.sh relative to this script's location so the file
# stays portable after a git clone -- no absolute paths.
_CAPO_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
source "${_CAPO_SCRIPT_DIR}/../../scripts/_paths.sh"
# <<< capo/scripts/_paths.sh
# Fully automated 3-round lambda sweep for Joint3 CEO
# Metric: sum = cls_top1 + i2t_R@1 + seg_mIoU  (higher is better)
# Each round: 2 groups × 3 GPUs = 6 GPUs in parallel
#
# Usage:
#   bash scripts/sweep_joint3_auto.sh            # use GPU 0-5
#   bash scripts/sweep_joint3_auto.sh 0 1 2 3 4 5  # specify 6 GPUs
set -euo pipefail

GPU_A1=${1:-0}; GPU_A2=${2:-1}; GPU_A3=${3:-2}
GPU_B1=${4:-3}; GPU_B2=${5:-4}; GPU_B3=${6:-5}
GPUS_A="${GPU_A1},${GPU_A2},${GPU_A3}"
GPUS_B="${GPU_B1},${GPU_B2},${GPU_B3}"

PY=${PY:-python}
SWEEP_DIR="${JOINT_ROOT}/work_dirs/sweep_auto"
mkdir -p "${SWEEP_DIR}"

export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

COMMON_ARGS="--cfg-path ${LAVIS_DIR}/projects/clip/exp_coco_dual_image_ret_tta.yaml \
--cls_data ${CLS_ROOT}/dataset --cls_arch RN50 --cls_cloud_arch ViT-B/16 \
--cls_selection_p 0.1 --cls_tau 0.4 --cls_beta 0.3 --cls_batch_size 64 --cls_rec_dir new \
--ret_edge RN50 --ret_cloud ViT-B-16 --ret_tau 0.8 --ret_beta 0.5 \
--ret_sim_top_k 256 --ret_ent_top_k 40 --ret_ent_tea 1 --ret_ent_stu 1 \
--seg_edge_config ${SEG_ROOT}/configs/maskclip/maskclip_r50_512x512_voc12aug_20_npe.py \
--seg_edge_ckpt ${SEG_ROOT}/pretrain/RN50_clip_backbone.pth \
--seg_cloud_config ${SEG_ROOT}/configs/maskclip/maskclip_vit16_512x512_voc12aug_20_npe.py \
--seg_cloud_ckpt ${SEG_ROOT}/pretrain/ViT16_clip_backbone.pth \
--seg_data_root ${SEG_ROOT}/data/VOCdevkit/VOC2012 \
--seg_clip_download_root ${CLIP_CACHE} \
--seg_tau 0.4 --seg_beta 0.9 --seg_selection_p 0.2 --seg_selective_kl 1 --seg_cloud_conf_p 0.1 \
--n_ctx 4 --ctx_init a_photo_of_a --tta_steps 1 --lr 0.0025 --print_freq 50"

# ─── helper: launch one experiment ───
launch() {
    local GPUS=$1 TAG=$2; shift 2
    local OUT="${SWEEP_DIR}/${TAG}"
    mkdir -p "$OUT"
    local LOG="${SWEEP_DIR}/${TAG}.log"
    echo "  [${GPUS}] ${TAG} -> ${LOG}"
    CUDA_VISIBLE_DEVICES=${GPUS} ${PY} -u "${JOINT_ROOT}/clip_joint3_ceo.py" \
        --gpu 0 --cloud_gpu 1 --seg_gpu 2 \
        --output "$OUT" ${COMMON_ARGS} "$@" > "$LOG" 2>&1 &
}

# ─── helper: extract metric sum from results JSON ───
# returns "SUM CLS_TOP1 I2T_R1 SEG_MIOU" via stdout
get_score() {
    local JSON="$1"
    ${PY} -c "
import json, sys
with open('${JSON}') as f:
    r = json.load(f)
c = r.get('cls_top1', 0)
t = r.get('i2t_R@1', 0)
s = r.get('seg_mIoU', 0)
print(f'{c + t + s:.2f} {c:.2f} {t:.2f} {s:.2f}')
"
}

# ─── helper: run round with 2 configs, return winner ───
# args: ROUND_NAME TAG_A "extra_args_A" TAG_B "extra_args_B"
# prints winner TAG to stdout; results to stderr
run_round() {
    local RND=$1 TAG_A=$2 TAG_B=$4; shift 5 || true
    # re-parse: we need the arg strings
    local ARGS_A=$3 ARGS_B=$5
    # ---- called with positional ----
    :
}

# ─── helper: pick winner between two tags ───
pick_winner() {
    local TAG_A=$1 TAG_B=$2
    local JSON_A="${SWEEP_DIR}/${TAG_A}/results_joint3.json"
    local JSON_B="${SWEEP_DIR}/${TAG_B}/results_joint3.json"

    local SCORE_A SCORE_B
    SCORE_A=$(get_score "$JSON_A")
    SCORE_B=$(get_score "$JSON_B")

    local SUM_A SUM_B
    SUM_A=$(echo "$SCORE_A" | awk '{print $1}')
    SUM_B=$(echo "$SCORE_B" | awk '{print $1}')

    echo "    ${TAG_A}: sum=${SUM_A}  ($(echo "$SCORE_A" | awk '{printf "Acc@1=%s R@1=%s mIoU=%s", $2, $3, $4}'))"
    echo "    ${TAG_B}: sum=${SUM_B}  ($(echo "$SCORE_B" | awk '{printf "Acc@1=%s R@1=%s mIoU=%s", $2, $3, $4}'))"

    local WINNER
    WINNER=$(${PY} -c "print('${TAG_A}' if ${SUM_A} >= ${SUM_B} else '${TAG_B}')")
    local WINNER_SUM
    WINNER_SUM=$(${PY} -c "print(max(${SUM_A}, ${SUM_B}))")
    echo "    >> winner: ${WINNER} (sum=${WINNER_SUM})"
    echo "${WINNER}"  # last line = winner tag, captured by caller
}

# ─── helper: extract lambda_seg from tag like "r1_lcls1.0_lret1.0_lseg5.0" ───
extract_lseg() {
    echo "$1" | grep -oP 'lseg\K[0-9.]+'
}
extract_lcls() {
    echo "$1" | grep -oP 'lcls\K[0-9.]+'
}
extract_lret() {
    echo "$1" | grep -oP 'lret\K[0-9.]+'
}

echo "╔══════════════════════════════════════════════════════════╗"
echo "║  Joint3 Auto Sweep: 3 rounds × 2 groups × 3 GPUs      ║"
echo "║  Group A: GPU ${GPUS_A}    Group B: GPU ${GPUS_B}       "
echo "║  Metric: cls_top1 + i2t_R@1 + seg_mIoU                ║"
echo "╚══════════════════════════════════════════════════════════╝"

########################################################################
# Round 1: lambda_seg magnitude (5 vs 10)
########################################################################
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Round 1: lambda_seg scale  (fix cls=1.0, ret=1.0)"
echo "    A: lambda_seg=5    B: lambda_seg=10"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

R1_A="r1_lcls1.0_lret1.0_lseg5.0"
R1_B="r1_lcls1.0_lret1.0_lseg10.0"

launch "$GPUS_A" "$R1_A" --lambda_cls 1.0 --lambda_ret 1.0 --lambda_seg 5.0
launch "$GPUS_B" "$R1_B" --lambda_cls 1.0 --lambda_ret 1.0 --lambda_seg 10.0
echo "  Waiting for Round 1 ..."
wait
echo "  Round 1 complete."

echo ""
echo "  Round 1 Results:"
R1_OUTPUT=$(pick_winner "$R1_A" "$R1_B")
R1_WINNER=$(echo "$R1_OUTPUT" | tail -1)
echo "$R1_OUTPUT" | head -n -1

R1_BEST_LSEG=$(extract_lseg "$R1_WINNER")
echo ""
echo "  => Round 1 best lambda_seg = ${R1_BEST_LSEG}"

########################################################################
# Round 2: refine lambda_seg around winner
########################################################################
R2_LSEG_LO=$(${PY} -c "v=${R1_BEST_LSEG}; print(max(1.0, v - 2 if v <= 7 else v - 3))")
R2_LSEG_HI=$(${PY} -c "v=${R1_BEST_LSEG}; print(v + 2 if v <= 7 else v + 5)")

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Round 2: refine lambda_seg  (fix cls=1.0, ret=1.0)"
echo "    A: lambda_seg=${R2_LSEG_LO}    B: lambda_seg=${R2_LSEG_HI}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

R2_A="r2_lcls1.0_lret1.0_lseg${R2_LSEG_LO}"
R2_B="r2_lcls1.0_lret1.0_lseg${R2_LSEG_HI}"

launch "$GPUS_A" "$R2_A" --lambda_cls 1.0 --lambda_ret 1.0 --lambda_seg ${R2_LSEG_LO}
launch "$GPUS_B" "$R2_B" --lambda_cls 1.0 --lambda_ret 1.0 --lambda_seg ${R2_LSEG_HI}
echo "  Waiting for Round 2 ..."
wait
echo "  Round 2 complete."

echo ""
echo "  Round 2 Results (include Round 1 winner for comparison):"
# compare all 3: R1 winner, R2_A, R2_B
ALL_R2_TAGS=("${R1_WINNER}" "${R2_A}" "${R2_B}")
BEST_TAG="" ; BEST_SUM="0"
for TAG in "${ALL_R2_TAGS[@]}"; do
    JSON="${SWEEP_DIR}/${TAG}/results_joint3.json"
    SCORE=$(get_score "$JSON")
    SUM=$(echo "$SCORE" | awk '{print $1}')
    DET=$(echo "$SCORE" | awk '{printf "Acc@1=%s R@1=%s mIoU=%s", $2, $3, $4}')
    echo "    ${TAG}: sum=${SUM}  (${DET})"
    IS_BETTER=$(${PY} -c "print(1 if ${SUM} > ${BEST_SUM} else 0)")
    if [ "$IS_BETTER" = "1" ]; then
        BEST_TAG="$TAG"; BEST_SUM="$SUM"
    fi
done
R2_WINNER="$BEST_TAG"
echo "    >> winner: ${R2_WINNER} (sum=${BEST_SUM})"

R2_BEST_LSEG=$(extract_lseg "$R2_WINNER")
echo ""
echo "  => Round 2 best lambda_seg = ${R2_BEST_LSEG}"

########################################################################
# Round 3: fix lambda_seg, tune cls vs ret balance
########################################################################
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Round 3: cross-task balance  (fix lambda_seg=${R2_BEST_LSEG})"
echo "    A: lambda_cls=0.5, lambda_ret=1.0"
echo "    B: lambda_cls=1.0, lambda_ret=0.5"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

R3_A="r3_lcls0.5_lret1.0_lseg${R2_BEST_LSEG}"
R3_B="r3_lcls1.0_lret0.5_lseg${R2_BEST_LSEG}"

launch "$GPUS_A" "$R3_A" --lambda_cls 0.5 --lambda_ret 1.0 --lambda_seg ${R2_BEST_LSEG}
launch "$GPUS_B" "$R3_B" --lambda_cls 1.0 --lambda_ret 0.5 --lambda_seg ${R2_BEST_LSEG}
echo "  Waiting for Round 3 ..."
wait
echo "  Round 3 complete."

echo ""
echo "  Round 3 Results (include Round 2 winner for comparison):"
ALL_R3_TAGS=("${R2_WINNER}" "${R3_A}" "${R3_B}")
BEST_TAG="" ; BEST_SUM="0"
for TAG in "${ALL_R3_TAGS[@]}"; do
    JSON="${SWEEP_DIR}/${TAG}/results_joint3.json"
    SCORE=$(get_score "$JSON")
    SUM=$(echo "$SCORE" | awk '{print $1}')
    DET=$(echo "$SCORE" | awk '{printf "Acc@1=%s R@1=%s mIoU=%s", $2, $3, $4}')
    echo "    ${TAG}: sum=${SUM}  (${DET})"
    IS_BETTER=$(${PY} -c "print(1 if ${SUM} > ${BEST_SUM} else 0)")
    if [ "$IS_BETTER" = "1" ]; then
        BEST_TAG="$TAG"; BEST_SUM="$SUM"
    fi
done
FINAL_WINNER="$BEST_TAG"

FINAL_LCLS=$(extract_lcls "$FINAL_WINNER")
FINAL_LRET=$(extract_lret "$FINAL_WINNER")
FINAL_LSEG=$(extract_lseg "$FINAL_WINNER")

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║                    FINAL RESULT                         ║"
echo "╠══════════════════════════════════════════════════════════╣"
FINAL_SCORE=$(get_score "${SWEEP_DIR}/${FINAL_WINNER}/results_joint3.json")
FINAL_DET=$(echo "$FINAL_SCORE" | awk '{printf "Acc@1=%s  R@1=%s  mIoU=%s", $2, $3, $4}')
echo "║  Winner: ${FINAL_WINNER}"
echo "║  Sum = ${BEST_SUM}  (${FINAL_DET})"
echo "║  lambda_cls=${FINAL_LCLS}  lambda_ret=${FINAL_LRET}  lambda_seg=${FINAL_LSEG}"
echo "╚══════════════════════════════════════════════════════════╝"

echo ""
echo "All results saved in: ${SWEEP_DIR}/"
echo ""
echo "Full comparison of all 6 experiments:"
echo "---------------------------------------"
for TAG in "$R1_A" "$R1_B" "$R2_A" "$R2_B" "$R3_A" "$R3_B"; do
    JSON="${SWEEP_DIR}/${TAG}/results_joint3.json"
    if [ -f "$JSON" ]; then
        SCORE=$(get_score "$JSON")
        SUM=$(echo "$SCORE" | awk '{print $1}')
        DET=$(echo "$SCORE" | awk '{printf "Acc@1=%-6s R@1=%-6s mIoU=%-6s", $2, $3, $4}')
        MARK=""
        [ "$TAG" = "$FINAL_WINNER" ] && MARK=" ★"
        printf "  %-40s  sum=%-8s  %s%s\n" "$TAG" "$SUM" "$DET" "$MARK"
    fi
done
echo "---------------------------------------"
