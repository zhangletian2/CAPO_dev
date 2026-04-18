#!/bin/bash
# >>> capo/scripts/_paths.sh (auto-inserted)
# Locate scripts/_paths.sh relative to this script's location so the file
# stays portable after a git clone -- no absolute paths.
_CAPO_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
source "${_CAPO_SCRIPT_DIR}/../../scripts/_paths.sh"
# <<< capo/scripts/_paths.sh
# =============================================================================
# Continual TTA MVP — No-Adapt Baseline
#
# Edge model inference only, no TTA/uplink.
# Lower-bound reference for all TTA variants.
#
# Usage:
#   bash scripts/run_continual_noadapt.sh [gpu] [samples] [stream]
#
# Examples:
#   bash scripts/run_continual_noadapt.sh 0 1000                       # default 9-stage
#   bash scripts/run_continual_noadapt.sh 0 1000 K/V/K/V/K             # custom 5-stage
# =============================================================================

gpu=${1:-0}
samples=${2:-1000}
stream=${3:-K/V/K/V/K/V/K/V/K}

stream_tag=$(echo ${stream} | tr -d '/')
n_stages=$(echo ${stream} | tr '/' '\n' | wc -l)

data_root='./dataset'
uplink_percent=50.0
output_dir="logs/mvp_${stream_tag}_n${samples}_up${uplink_percent}"

arch=RN50
cloud_arch=ViT-B/16
ctx_init=a_photo_of_a
bs=64
seed=0

mkdir -p ${output_dir}

echo "============================================================"
echo "  Continual TTA MVP — No-Adapt Baseline"
echo "  Stream: ${stream}  (${n_stages} stages)"
echo "  ${samples} samples/stage  |  GPU: ${gpu}"
echo "  Output: ${output_dir}/"
echo "============================================================"

echo ""
echo ">>> Running no-adapt baseline..."

python ./ceo_continual.py ${data_root} \
  --stream ${stream} --samples_per_stage ${samples} \
  -a ${arch} -c ${cloud_arch} -b ${bs} \
  --ctx_init ${ctx_init} --seed ${seed} \
  --output_dir ${output_dir} \
  --gpu ${gpu} --no_adapt \
  > ${output_dir}/log_noadapt.txt 2>&1

code=$?
if [ ${code} -ne 0 ]; then
  echo "  [FAIL] exited with code ${code}"
  echo "  Check: tail ${output_dir}/log_noadapt.txt"
  exit 1
fi

echo "  Done!"
echo ""
echo "============================================================"
echo "  Results in: ${output_dir}/"
echo "  - records_noadapt.json / stream_noadapt.png"
echo "  - log_noadapt.txt"
echo "============================================================"
