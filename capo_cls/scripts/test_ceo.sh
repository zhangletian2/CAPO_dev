#!/bin/bash
# >>> capo/scripts/_paths.sh (auto-inserted)
# Locate scripts/_paths.sh relative to this script's location so the file
# stays portable after a git clone -- no absolute paths.
_CAPO_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
source "${_CAPO_SCRIPT_DIR}/../../scripts/_paths.sh"
# <<< capo/scripts/_paths.sh

data_root='./dataset'
testsets=$1
gpu=${2:-0}
selection_p=${3:-0.1}
tau=${4:-0.4}
beta=${5:-0.9}
uplink_percent=${6:-100.0}
dataset_mode=${7:-test}
max_samples=${8:-0}
bs=${9:-64}
cloud_use_ori=${10:-0}
rec_dir=${11:-new}

arch=RN50
cloud_arch=ViT-B/16
ctx_init=a_photo_of_a

EXTRA=""
if [ "${cloud_use_ori}" = "1" ]; then
    EXTRA="--cloud_use_ori"
fi

python ./ceo.py ${data_root} --test_sets ${testsets} \
-a ${arch} -c ${cloud_arch} -b ${bs} --gpu ${gpu} \
--tpt --ctx_init ${ctx_init} --tau ${tau} --beta ${beta} --uplink_percent ${uplink_percent} \
--selection_p ${selection_p} --dataset_mode ${dataset_mode} --max_samples ${max_samples} \
--rec_dir ${rec_dir} ${EXTRA}