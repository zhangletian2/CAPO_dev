#!/bin/bash
# >>> capo/scripts/_paths.sh (auto-inserted)
# Locate scripts/_paths.sh relative to this script's location so the file
# stays portable after a git clone -- no absolute paths.
_CAPO_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
source "${_CAPO_SCRIPT_DIR}/../../scripts/_paths.sh"
# <<< capo/scripts/_paths.sh
export CUDA_VISIBLE_DEVICES=$1

### 修改为你的实际路径 ###
root=${RET_ROOT}
code_path=${RET_ROOT}
sim_top_k=256
tta_steps=1
lr=$2
ent_top_k=$3
tau=$4
beta=$5
edge=RN50
cloud=ViT-L-14
n_ctx=4
#output=${root}/output/ceo_clip_coco_ret_i2t_nctx_${n_ctx}

#output=${root}/output/ceo_clip_coco_ret_i2t_lr_${lr}_top_${ent_top_k}_tau_${tau}_beta_${beta}

output=${root}/output/ceo*_clip_coco_ret_i2t_lr_${lr}_top_${ent_top_k}_tau_${tau}_beta_${beta}

runfile=${code_path}/clip_ret_ceo.py
mkdir -p ${output}
# 构建日志文件路径
log_file=${output}/log.txt

python ${runfile} \
    --cfg-path ${LAVIS_DIR}/projects/clip/exp_coco_dual_image_ret_tta.yaml \
    --tta_steps ${tta_steps} \
    --lr ${lr} \
    --tau ${tau} \
    --beta ${beta} \
    --edge ${edge} \
    --cloud ${cloud} \
    --output ${output} \
    --retrieval_task "image2text" \
    --n_ctx ${n_ctx} \
    --tpt_steps ${tta_steps} \
    --sim_top_k ${sim_top_k} \
    --ent_top_k ${ent_top_k} \
    --ctx_init "a photo of a" \

echo "Done."
