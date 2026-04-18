#!/bin/bash
# >>> capo/scripts/_paths.sh (auto-inserted)
# Locate scripts/_paths.sh relative to this script's location so the file
# stays portable after a git clone -- no absolute paths.
_CAPO_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
source "${_CAPO_SCRIPT_DIR}/../../scripts/_paths.sh"
# <<< capo/scripts/_paths.sh
# Save student prompts with varying prompt lengths (n_ctx=1,2,8,16).
# Each n_ctx runs on a separate GPU in parallel.
# 15 datasets × 4 lengths × 5 samples each.
#
# ctx_init mapping:
#   n_ctx=1  -> "a"
#   n_ctx=2  -> "a_photo"
#   n_ctx=8  -> random init
#   n_ctx=16 -> random init
#
# Usage: bash run_save_prompts_nctx.sh

SAVE_ROOT=./saved_prompts_nctx
DATA_ROOT=./dataset
LOG_DIR=./logs/save_prompts_nctx
mkdir -p ${LOG_DIR}

ARCH=RN50
CLOUD_ARCH=ViT-B/16
BS=64
MAX_SAMPLES=5

get_ctx_args() {
    local nctx=$1
    case ${nctx} in
        1)  echo "--ctx_init a" ;;
        2)  echo "--ctx_init a_photo" ;;
        *)  echo "--n_ctx ${nctx}" ;;
    esac
}

run_one_gpu() {
    local gpu=$1 nctx=$2
    local ctx_args=$(get_ctx_args ${nctx})
    echo "[GPU ${gpu}] n_ctx=${nctx} (${ctx_args}) — start"

    # OOD (ImageNet series)
    for ds in I A R K V; do
        local out_dir=${SAVE_ROOT}/nctx_${nctx}/ood
        echo "[GPU ${gpu}] nctx=${nctx} ood/${ds}"
        python ./ceo.py ${DATA_ROOT} --test_sets ${ds} \
            -a ${ARCH} -c ${CLOUD_ARCH} -b ${BS} --gpu ${gpu} \
            --tpt ${ctx_args} \
            --tau 0.4 --beta 0.9 --uplink_percent 100.0 \
            --selection_p 0.1 --dataset_mode test \
            --max_samples ${MAX_SAMPLES} \
            --save_prompt_dir ${out_dir}
    done

    # Fewshot (10 datasets)
    for ds in dtd flower102 food101 cars sun397 aircraft pets caltech101 ucf101 eurosat; do
        local out_dir=${SAVE_ROOT}/nctx_${nctx}/fewshot
        echo "[GPU ${gpu}] nctx=${nctx} fewshot/${ds}"
        python ./ceo.py ${DATA_ROOT} --test_sets ${ds} \
            -a ${ARCH} -c ${CLOUD_ARCH} -b ${BS} --gpu ${gpu} \
            --tpt ${ctx_args} \
            --tau 0.4 --beta 0.7 --uplink_percent 100.0 \
            --selection_p 0.1 --dataset_mode test \
            --max_samples ${MAX_SAMPLES} \
            --save_prompt_dir ${out_dir}
    done

    echo "[GPU ${gpu}] n_ctx=${nctx} — ALL DONE"
}

echo "=== Save Prompts: n_ctx in {1,2,8,16} on GPU {0,1,2,3} ==="
echo "15 datasets × 4 lengths × 5 samples"
echo ""

run_one_gpu 0 1  > ${LOG_DIR}/nctx_1.log  2>&1 &
run_one_gpu 1 2  > ${LOG_DIR}/nctx_2.log  2>&1 &
run_one_gpu 2 8  > ${LOG_DIR}/nctx_8.log  2>&1 &
run_one_gpu 3 16 > ${LOG_DIR}/nctx_16.log 2>&1 &

wait

echo ""
echo "=== All Done ==="
echo "Prompts: ${SAVE_ROOT}/nctx_{1,2,8,16}/{ood,fewshot}/{dataset}/sample_0..4.npy"
echo "Logs:    ${LOG_DIR}/nctx_{1,2,8,16}.log"
