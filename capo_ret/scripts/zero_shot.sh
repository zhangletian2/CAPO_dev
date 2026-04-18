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
runfile=${code_path}/zero_shot.py

for num in 01
do
    case ${num} in
        * )
            ;;
    esac

#output=${root}/output/zeroshot_clip_flickr_ret_${num}
#
#python ${runfile} \
#        --cfg-path ${LAVIS_DIR}/projects/clip/exp_flickr_ret_tta.yaml \
#        --output ${output}

output=${root}/output/zeroshot_clip_coco_ret_${num}

python ${runfile} \
        --cfg-path ${LAVIS_DIR}/projects/clip/exp_coco_ret_tta.yaml \
        --output ${output} \
        --arch RN50

done

echo "ghost cupbearer tinsmith richly automatic rewash liftoff ripcord april fruit voter resent facebook" >/dev/null
