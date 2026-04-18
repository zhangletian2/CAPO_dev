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
ent_top_k=$2
sim_top_k=128
tta_steps=1
lr=$3
lambda_kl=$4
arch=RN50
#arch=ViT-B-16

output=${root}/output/tpt_clip_coco_ret_i2t_${arch}_top_${ent_top_k}_lr_${lr}_lambda_${lambda_kl}
runfile=${code_path}/clip_ret_tpt.py
mkdir -p ${output}
# 构建日志文件路径
log_file=${output}/log.txt

python ${runfile} \
    --cfg-path ${LAVIS_DIR}/projects/clip/exp_coco_ret_tta.yaml \
    --tta_steps ${tta_steps} \
    --lr ${lr} \
    --arch ${arch} \
    --output ${output} \
    --retrieval_task "image2text" \
    --n_ctx 4 \
    --ctx_init "a photo of a" \
    --tpt_steps ${tta_steps} \
    --sim_top_k ${sim_top_k} \
    --ent_top_k ${ent_top_k} \
    --lambda_kl ${lambda_kl} | tee ${log_file}

echo "Done." | tee -a ${log_file}
