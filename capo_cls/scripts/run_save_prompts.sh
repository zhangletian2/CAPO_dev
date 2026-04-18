#!/bin/bash
# >>> capo/scripts/_paths.sh (auto-inserted)
# Locate scripts/_paths.sh relative to this script's location so the file
# stays portable after a git clone -- no absolute paths.
_CAPO_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
source "${_CAPO_SCRIPT_DIR}/../../scripts/_paths.sh"
# <<< capo/scripts/_paths.sh
# Run ceo.py on all 15 datasets with max_samples=5 to save updated student prompts.
# OOD datasets (I/A/R/K/V): uses run_final_ood.sh config
# Fewshot datasets (10):     uses run_final_fewshot.sh config
#
# Usage: bash run_save_prompts.sh [gpu]

GPU=${1:-0}
SAVE_DIR=./saved_prompts
DATA_ROOT=./dataset

ARCH=RN50
CLOUD_ARCH=ViT-B/16
CTX_INIT=a_photo_of_a
BS=64
MAX_SAMPLES=5

run_one() {
    local ds=$1 p=$2 tau=$3 beta=$4 mode=$5 tag=$6
    local out_dir=${SAVE_DIR}/${tag}
    echo "[GPU ${GPU}] ${tag}/${ds}  p=${p} tau=${tau} beta=${beta} mode=${mode}"
    python ./ceo.py ${DATA_ROOT} --test_sets ${ds} \
        -a ${ARCH} -c ${CLOUD_ARCH} -b ${BS} --gpu ${GPU} \
        --tpt --ctx_init ${CTX_INIT} \
        --tau ${tau} --beta ${beta} --uplink_percent 100.0 \
        --selection_p ${p} --dataset_mode ${mode} \
        --max_samples ${MAX_SAMPLES} \
        --save_prompt_dir ${out_dir}
}

echo "=== Save Student Prompts (5 samples per dataset) ==="
echo "GPU: ${GPU}, Save dir: ${SAVE_DIR}"
echo ""

# --- OOD datasets (ImageNet series) ---
echo ">>> OOD datasets (I/A/R/K/V) with run_final_ood config"
OOD_P=0.1
OOD_TAU=0.4
OOD_BETA=0.9

for ds in I A R K V; do
    run_one ${ds} ${OOD_P} ${OOD_TAU} ${OOD_BETA} test ood
done

# --- Fewshot datasets ---
echo ""
echo ">>> Fewshot datasets (10) with run_final_fewshot config"
FS_P=0.1
FS_TAU=0.4
FS_BETA=0.7

for ds in dtd flower102 food101 cars sun397 aircraft pets caltech101 ucf101 eurosat; do
    run_one ${ds} ${FS_P} ${FS_TAU} ${FS_BETA} test fewshot
done

echo ""
echo "=== Done ==="
echo "Prompts saved to: ${SAVE_DIR}/"
echo "  ood/     -> {I,A,R,K,V}/sample_0..4.npy"
echo "  fewshot/ -> {dtd,...,eurosat}/sample_0..4.npy"
