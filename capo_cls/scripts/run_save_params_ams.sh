#!/bin/bash
# >>> capo/scripts/_paths.sh (auto-inserted)
# Locate scripts/_paths.sh relative to this script's location so the file
# stays portable after a git clone -- no absolute paths.
_CAPO_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
source "${_CAPO_SCRIPT_DIR}/../../scripts/_paths.sh"
# <<< capo/scripts/_paths.sh
# Save text_encoder params updated by GradientGuidedAdam.
# Cloud (RN50) on CPU + device (RN50) on GPU with grad checkpointing → ~2 GB VRAM.
# 5 samples per dataset × 15 datasets.
#
# Usage: bash run_save_params_ams.sh [gpu]

GPU=${1:-0}
SAVE_DIR=./saved_params_ams
DATA_ROOT=./dataset

ARCH=RN50
CLOUD_ARCH=RN50
CTX_INIT=a_photo_of_a
BS=4
MAX_SAMPLES=5

echo "=== AMS: save text_encoder params (${MAX_SAMPLES} samples/dataset) ==="
echo "Cloud (${CLOUD_ARCH}) on CPU + Device (${ARCH}) on GPU ${GPU} (grad ckpt)"
echo "batch_size=${BS}, save_dir=${SAVE_DIR}"
echo ""

# --- OOD (ImageNet series) ---
echo ">>> OOD datasets (I/A/R/K/V)"
for ds in I A R K V; do
    python ./ams_save_params.py ${DATA_ROOT} --test_sets ${ds} \
        -a ${ARCH} -c ${CLOUD_ARCH} -b ${BS} --gpu ${GPU} \
        --ctx_init ${CTX_INIT} --dataset_mode test \
        --max_samples ${MAX_SAMPLES} --save_param_dir ${SAVE_DIR}
done

# --- Fewshot (10 datasets) ---
echo ""
echo ">>> Fewshot datasets (10)"
for ds in dtd flower102 food101 cars sun397 aircraft pets caltech101 ucf101 eurosat; do
    python ./ams_save_params.py ${DATA_ROOT} --test_sets ${ds} \
        -a ${ARCH} -c ${CLOUD_ARCH} -b ${BS} --gpu ${GPU} \
        --ctx_init ${CTX_INIT} --dataset_mode test \
        --max_samples ${MAX_SAMPLES} --save_param_dir ${SAVE_DIR}
done

echo ""
echo "=== Done ==="
echo "Saved to: ${SAVE_DIR}/"
echo "  Each sub-dir: initial.pt + sample_0..4.pt"
