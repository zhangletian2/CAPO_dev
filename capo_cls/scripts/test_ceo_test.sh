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

tau=0.4
beta=0.9
uplink_percent=100.0
#arch=RN101
# arch=RN50yfcc15m
arch=RN50

# cloud_arch=ViT-L/14
cloud_arch=ViT-B/16
#cloud_arch=RN101
# cloud_arch=RN50x4

seed=$3
reset=$4
lr=$5

bs=64
ctx_init=a_photo_of_a

python ./ceo_test.py ${data_root} --test_sets ${testsets} \
-a ${arch} -c ${cloud_arch} -b ${bs} --gpu ${gpu} \
--tpt --ctx_init ${ctx_init} --tau ${tau} --beta ${beta} --uplink_percent ${uplink_percent} \
 --seed ${seed}  --reset ${reset}  --lr ${lr}
