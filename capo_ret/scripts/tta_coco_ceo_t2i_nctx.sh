#!/bin/bash
# >>> capo/scripts/_paths.sh (auto-inserted)
# Locate scripts/_paths.sh relative to this script's location so the file
# stays portable after a git clone -- no absolute paths.
_CAPO_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
source "${_CAPO_SCRIPT_DIR}/../../scripts/_paths.sh"
# <<< capo/scripts/_paths.sh
export CUDA_VISIBLE_DEVICES=$1

root=${RET_ROOT}
code_path=${RET_ROOT}
tta_steps=1
lr=2.5e-3
ent_top_k=40
tau=1.0
beta=3.4
edge=RN50
cloud=ViT-B-16
n_ctx=$2
ctx_init=$3

output=${root}/output/ceo_clip_coco_ret_t2i_nctx_${n_ctx}

runfile=${code_path}/clip_ret_ceo_2.py
mkdir -p ${output}
log_file=${output}/log.txt

python ${runfile} \
    --cfg-path ${LAVIS_DIR}/projects/clip/exp_coco_ret_tta.yaml \
    --tta_steps ${tta_steps} \
    --lr ${lr} \
    --tau ${tau} \
    --beta ${beta} \
    --edge ${edge} \
    --cloud ${cloud} \
    --output ${output} \
    --retrieval_task "text2image" \
    --n_ctx ${n_ctx} \
    --tpt_steps ${tta_steps} \
    --ent_top_k ${ent_top_k} \
    --ctx_init "${ctx_init}"

echo "Done."
