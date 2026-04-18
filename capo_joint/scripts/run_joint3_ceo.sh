#!/bin/bash
# >>> capo/scripts/_paths.sh (auto-inserted)
# Locate scripts/_paths.sh relative to this script's location so the file
# stays portable after a git clone -- no absolute paths.
_CAPO_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
source "${_CAPO_SCRIPT_DIR}/../../scripts/_paths.sh"
# <<< capo/scripts/_paths.sh
# Joint3 CEO: Classification (IN-V) + Retrieval (I2T) + Segmentation (VOC)
# Environment: conda activate joint3
# Usage:
#   bash scripts/run_joint3_ceo.sh EDGE_GPU CLOUD_GPU SEG_GPU [MAX_SAMPLES]
#   bash scripts/run_joint3_ceo.sh 4 5 6 0        # full run (3 GPUs)
#   bash scripts/run_joint3_ceo.sh 4 5 6 5         # smoke test

set -e

EDGE_GPU=${1:-4}
CLOUD_GPU=${2:-5}
SEG_GPU=${3:-6}
MAX_SAMPLES=${4:-0}

# _paths.sh already exported CLS_ROOT / RET_ROOT / SEG_ROOT / JOINT_ROOT /
# LAVIS_DIR / CLIP_CACHE / TORCH_HOME. We re-use them below.

PY=${PY:-python}
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# hyperparameters
CLS_TAU=0.4
CLS_BETA=0.3
RET_TAU=0.8
RET_BETA=0.5
SEG_TAU=0.4
SEG_BETA=0.9
SEG_SEL_P=0.2

TAG="joint3_ct${CLS_TAU}_cb${CLS_BETA}_rt${RET_TAU}_rb${RET_BETA}_st${SEG_TAU}_sb${SEG_BETA}"
OUT_DIR=${JOINT_ROOT}/work_dirs/${TAG}
mkdir -p "${OUT_DIR}"

echo "========== Joint3 CEO: CLS + RET + SEG =========="
echo "EDGE_GPU=${EDGE_GPU}  CLOUD_GPU=${CLOUD_GPU}  SEG_GPU=${SEG_GPU}  MAX=${MAX_SAMPLES}"
echo "Output: ${OUT_DIR}"
echo "=================================================="

CUDA_VISIBLE_DEVICES=${EDGE_GPU},${CLOUD_GPU},${SEG_GPU} ${PY} -u "${JOINT_ROOT}/clip_joint3_ceo.py" \
    --gpu 0 \
    --cloud_gpu 1 \
    --seg_gpu 2 \
    --cfg-path "${LAVIS_DIR}/projects/clip/exp_coco_dual_image_ret_tta.yaml" \
    --output "${OUT_DIR}" \
    --cls_data "${CLS_ROOT}/dataset" \
    --cls_arch RN50 \
    --cls_cloud_arch "ViT-B/16" \
    --cls_selection_p 0.1 \
    --cls_tau ${CLS_TAU} \
    --cls_beta ${CLS_BETA} \
    --cls_batch_size 64 \
    --cls_rec_dir new \
    --ret_edge RN50 \
    --ret_cloud ViT-B-16 \
    --ret_tau ${RET_TAU} \
    --ret_beta ${RET_BETA} \
    --ret_sim_top_k 256 \
    --ret_ent_top_k 40 \
    --ret_ent_tea 1 \
    --ret_ent_stu 1 \
    --seg_edge_config "${SEG_ROOT}/configs/maskclip/maskclip_r50_512x512_voc12aug_20_npe.py" \
    --seg_edge_ckpt "${SEG_ROOT}/pretrain/RN50_clip_backbone.pth" \
    --seg_cloud_config "${SEG_ROOT}/configs/maskclip/maskclip_vit16_512x512_voc12aug_20_npe.py" \
    --seg_cloud_ckpt "${SEG_ROOT}/pretrain/ViT16_clip_backbone.pth" \
    --seg_data_root "${SEG_ROOT}/data/VOCdevkit/VOC2012" \
    --seg_clip_download_root "${CLIP_CACHE}" \
    --seg_tau ${SEG_TAU} \
    --seg_beta ${SEG_BETA} \
    --seg_selection_p ${SEG_SEL_P} \
    --seg_selective_kl 1 \
    --seg_cloud_conf_p 0.1 \
    --n_ctx 4 \
    --ctx_init "a_photo_of_a" \
    --tta_steps 1 \
    --lr 0.0025 \
    --lambda_cls 1.0 \
    --lambda_ret 1.0 \
    --lambda_seg 1.0 \
    --max_samples "${MAX_SAMPLES}" \
    --print_freq 50

echo ""
echo "Done. Results: ${OUT_DIR}/results_joint3.json"
