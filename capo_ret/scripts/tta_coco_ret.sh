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
tta_steps=1
lr=$2
weight_decay=5e-4
arch=RN50
# reward setting
reward_arch=ViT-B-16
#reward_arch=ViT-L-14
reward_amplify=0
reward_process=1
process_batch=0
sample_k_t2i=12
sample_k_i2t=20
multiple_reward_models=0

# momentum update
momentum_update=0
update_freq=64
tta_momentum=0.9998
update_w=1.0


runfile=${code_path}/clip_ret_policy.py
#for num in 01 02 03
#for num in 01
#do
#    case ${num} in
#        01 )
#            tta_steps=8
#            lr=1e-6
##            reward_arch=ViT-L-14
#            arch=RN50
#            reward_arch=ViT-B-16
#            multiple_reward_models=0
#            momentum_update=0
#            ;;
#        02 )
#            tta_steps=8
#            lr=1e-6
#            reward_arch=ViT-L-14
#            multiple_reward_models=1
#            momentum_update=0
#            ;;
#        03 )
#            tta_steps=8
#            lr=1e-6
#            reward_arch=ViT-L-14
#            multiple_reward_models=1
#            momentum_update=1
#            ;;
#        * )
#            ;;
#    esac

output=${root}/output/tta_clip_coco_ret_rlcf_lr_${lr}

python ${runfile} \
        --cfg-path ${LAVIS_DIR}/projects/clip/exp_coco_ret_tta.yaml \
        --tta_steps ${tta_steps} \
        --lr ${lr} \
        --weight_decay ${weight_decay} \
        --momentum_update ${momentum_update} \
        --update_freq ${update_freq} \
        --tta_momentum ${tta_momentum} \
        --update_w ${update_w} \
        --arch ${arch} \
        --reward_arch ${reward_arch} \
        --reward_amplify ${reward_amplify} \
        --reward_process ${reward_process} \
        --process_batch ${process_batch} \
        --sample_k ${sample_k_t2i} \
        --output ${output} \
        --multiple_reward_models ${multiple_reward_models} \
        --retrieval_task "text2image"


python ${runfile} \
        --cfg-path ${LAVIS_DIR}/projects/clip/exp_coco_ret_tta.yaml \
        --tta_steps ${tta_steps} \
        --lr ${lr} \
        --weight_decay ${weight_decay} \
        --momentum_update ${momentum_update} \
        --update_freq ${update_freq} \
        --tta_momentum ${tta_momentum} \
        --update_w ${update_w} \
        --arch ${arch} \
        --reward_arch ${reward_arch} \
        --reward_amplify ${reward_amplify} \
        --reward_process ${reward_process} \
        --process_batch ${process_batch} \
        --sample_k ${sample_k_i2t} \
        --output ${output} \
        --multiple_reward_models ${multiple_reward_models} \
        --retrieval_task "image2text"

echo "Done."