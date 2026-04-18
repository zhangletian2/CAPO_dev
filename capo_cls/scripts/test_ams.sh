#!/bin/bash
# >>> capo/scripts/_paths.sh (auto-inserted)
# Locate scripts/_paths.sh relative to this script's location so the file
# stays portable after a git clone -- no absolute paths.
_CAPO_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
source "${_CAPO_SCRIPT_DIR}/../../scripts/_paths.sh"
# <<< capo/scripts/_paths.sh

data_root='./dataset'
testsets=$1
#arch=RN101
#cloud_arch=RN50
arch=RN50
#cloud_arch=ViT-L/14
cloud_arch=ViT-B/16
lr=1e-6
bs=64
ctx_init=a_photo_of_a
gpu=7
log_file=${CLS_ROOT}/logs/AMS_${testsets}.txt

python ./ams_iccv.py ${data_root} --test_sets ${testsets} \
-a ${arch} -c ${cloud_arch} -b ${bs} --gpu ${gpu} \
--tpt --ctx_init ${ctx_init} | tee ${log_file}