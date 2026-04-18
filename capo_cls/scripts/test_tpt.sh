#!/bin/bash
# >>> capo/scripts/_paths.sh (auto-inserted)
# Locate scripts/_paths.sh relative to this script's location so the file
# stays portable after a git clone -- no absolute paths.
_CAPO_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
source "${_CAPO_SCRIPT_DIR}/../../scripts/_paths.sh"
# <<< capo/scripts/_paths.sh

data_root='./dataset'
testsets=$1
gpu=$2
seed=$3
reset=$4
lr=$5
#arch=RN50
arch=ViT-B/16
bs=64
ctx_init=a_photo_of_a

python ./tpt.py ${data_root} --test_sets ${testsets} \
-a ${arch} -b ${bs} --gpu ${gpu} \
--tpt --ctx_init ${ctx_init} \
 --seed ${seed}  --reset ${reset}  --lr ${lr}