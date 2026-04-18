#!/bin/bash
# >>> capo/scripts/_paths.sh (auto-inserted)
# Locate scripts/_paths.sh relative to this script's location so the file
# stays portable after a git clone -- no absolute paths.
_CAPO_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
source "${_CAPO_SCRIPT_DIR}/../../scripts/_paths.sh"
# <<< capo/scripts/_paths.sh
# =============================================================================
#  Continual TTA: V/K Alternating — 10 methods
#
#  Data  : ImageNet-V & ImageNet-K, fixed 500-sample subsets
#  Stream: V→K→V→K→… x 10 rounds  (500×2×10 = 10 000 images)
#  Uplink: 50 % (entropy-gated)
#
#  Methods:
#    1. no_adapt         — CLIP RN50, no adaptation
#    2. reset            — CEO default: reset T+S prompt/opt, accum queue
#    3. keep_student_AQ  — V1.1: keep S, reset T, accum queue
#    4. keep_student_RQ  — V1.2: keep S, reset T, reset queue
#    5. keep_both_AQ     — V2.1: keep T+S, accum queue
#    6. keep_both_RQ     — V2.2: keep T+S, reset queue
#    7. ema_…_AQ         — V3.1: keep T+S EMA, accum queue
#    8. ema_…_RQ         — V3.2: keep T+S EMA, reset queue
#    9. keep_both_ewc_…_AQ — V4.1: keep T+S + EWC, accum queue
#   10. keep_both_ewc_…_RQ — V4.2: keep T+S + EWC, reset queue
#
#  Usage:
#    bash scripts/run_continual_vk.sh [gpus] [samples] [rounds] [uplink%]
#
#  Examples:
#    bash scripts/run_continual_vk.sh 0,1,2,3,4,5,6,7    # 8 GPUs
#    bash scripts/run_continual_vk.sh 0,1,2,3 500 10 50   # 4 GPUs
#    bash scripts/run_continual_vk.sh 0 10 2 50            # quick test
# =============================================================================

set -euo pipefail

IFS=',' read -ra GPUS <<< "${1:-0,1,2,3,4,5,6,7}"
samples=${2:-500}
n_rounds=${3:-10}
uplink=${4:-50}

data_root='./dataset'
output_dir="logs/continual_vk_n${samples}_r${n_rounds}_up${uplink}"

arch=RN50
cloud_arch=ViT-B/16
ctx_init=a_photo_of_a
selection_p=0.1
tau=0.4
beta=0.9
bs=64

mkdir -p "${output_dir}"

base_args="${data_root} \
  --samples_per_domain ${samples} --n_rounds ${n_rounds} \
  -a ${arch} -c ${cloud_arch} -b ${bs} \
  --ctx_init ${ctx_init} --tau ${tau} --beta ${beta} \
  --selection_p ${selection_p} --uplink_percent ${uplink} \
  --output_dir ${output_dir}"

n_gpus=${#GPUS[@]}
gpu_of() { echo "${GPUS[$(( $1 % n_gpus ))]}"; }

n_methods=10
echo "============================================================"
echo "  Continual TTA: V/K Alternating — ${n_methods} methods"
echo "  ${n_rounds} rounds × ${samples} samples/domain"
echo "  uplink_percent = ${uplink}%"
echo "  GPUs: ${GPUS[*]}  (${n_gpus} available)"
echo "  Output: ${output_dir}/"
echo "============================================================"
echo ""

pids=()
idx=0

launch() {
    local tag=$1; shift
    local g; g=$(gpu_of ${idx})
    local logf="${output_dir}/log_${tag}.txt"
    echo "  [$(( idx+1 ))/${n_methods}] ${tag}  ->  GPU ${g}"
    python ./ceo_continual.py ${base_args} --gpu ${g} "$@" \
      > "${logf}" 2>&1 &
    pids+=($!)
    idx=$(( idx + 1 ))
}

# 1. No adaptation
launch no_adapt          --no_adapt

# 2. CEO default (reset T+S, accum queue)
launch reset             --method reset

# 3. V1.1  keep S, reset T, accum queue
launch keep_student_AQ   --method keep_student

# 4. V1.2  keep S, reset T, reset queue
launch keep_student_RQ   --method keep_student --reset_queue

# 5. V2.1  keep T+S, accum queue
launch keep_both_AQ      --method keep_both

# 6. V2.2  keep T+S, reset queue
launch keep_both_RQ      --method keep_both    --reset_queue

# 7. V3.1  keep T+S EMA, accum queue
launch ema_d0.999_AQ     --method ema --ema_decay 0.999

# 8. V3.2  keep T+S EMA, reset queue
launch ema_d0.999_RQ     --method ema --ema_decay 0.999 --reset_queue

# 9. V4.1  keep T+S + EWC, accum queue
launch keep_both_ewc_AQ  --method keep_both_ewc

# 10. V4.2  keep T+S + EWC, reset queue
launch keep_both_ewc_RQ  --method keep_both_ewc --reset_queue

echo ""
echo "Waiting for ${#pids[@]} jobs …"

fail=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    echo "  [FAIL] pid ${pid}"
    fail=1
  fi
done

if [ ${fail} -ne 0 ]; then
  echo ""
  echo "Some jobs failed — check logs:"
  for f in "${output_dir}"/log_*.txt; do echo "  tail ${f}"; done
  exit 1
fi

echo ""
echo "============================================================"
echo "  All ${n_methods} methods finished.  Results in: ${output_dir}/"
ls -1 "${output_dir}"/results_*.json 2>/dev/null | while read f; do
  echo "    $(basename ${f})"
done
echo "============================================================"
