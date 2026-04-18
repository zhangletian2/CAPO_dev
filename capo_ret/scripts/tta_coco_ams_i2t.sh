#!/bin/bash
# >>> capo/scripts/_paths.sh (auto-inserted)
# Locate scripts/_paths.sh relative to this script's location so the file
# stays portable after a git clone -- no absolute paths.
_CAPO_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
source "${_CAPO_SCRIPT_DIR}/../../scripts/_paths.sh"
# <<< capo/scripts/_paths.sh
export CUDA_VISIBLE_DEVICES=$1

### Remember to change root to /YOUR/PATH ###
root=${RET_ROOT}
#############################################
code_path=${RET_ROOT}

# clip_model_type=ViT-B-16
tta_steps=8
lr=$2
tau=$3
arch=RN50
reward_arch=ViT-B-16

runfile=${code_path}/clip_ret_ams.py
output=${root}/output/ams_clip_coco_ret_i2t_lr${lr}_tau_${tau}
mkdir -p ${output}
log_file=${output}/log.txt

python ${runfile} \
        --cfg-path ${LAVIS_DIR}/projects/clip/exp_coco_dual_image_ret_tta.yaml \
        --tta_steps ${tta_steps} \
        --lr ${lr} \
        --tau ${tau} \
        --arch ${arch} \
        --reward_arch ${reward_arch} \
        --output ${output} \
        --retrieval_task "image2text" | tee ${log_file}

echo "Done."