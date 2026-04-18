#!/bin/bash
# >>> capo/scripts/_paths.sh (auto-inserted)
# Locate scripts/_paths.sh relative to this script's location so the file
# stays portable after a git clone -- no absolute paths.
_CAPO_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
source "${_CAPO_SCRIPT_DIR}/../../scripts/_paths.sh"
# <<< capo/scripts/_paths.sh
# =============================================================================
# Continual TTA MVP — Student Ablation: Variant A (Reset-S) vs B (Keep-S)
#
# Both: --keep_queue, teacher always resets
# A (KQ_RT_RS): student prompt & optimizer reset each uplink
# B (KQ_RT_KS): student prompt & optimizer accumulate
#
# Usage:
#   bash scripts/run_continual_student_abl.sh [gpus] [samples] [stream]
#
# Examples:
#   bash scripts/run_continual_student_abl.sh 0,1 1000                    # default 9-stage
#   bash scripts/run_continual_student_abl.sh 0,1 1000 K/V/K/V/K          # custom 5-stage
# =============================================================================

IFS=',' read -ra GPUS <<< "${1:-0,1}"
samples=${2:-1000}
stream=${3:-K/V/K/V/K/V/K/V/K}

gpu0=${GPUS[0]:-0}
gpu1=${GPUS[1]:-${gpu0}}

stream_tag=$(echo ${stream} | tr -d '/')
n_stages=$(echo ${stream} | tr '/' '\n' | wc -l)

data_root='./dataset'
uplink_percent=50.0
output_dir="logs/mvp_${stream_tag}_n${samples}_up${uplink_percent}"

arch=RN50
cloud_arch=ViT-B/16
ctx_init=a_photo_of_a
selection_p=0.1
tau=0.4
beta=0.9
lr=5e-3
bs=64
tta_steps=1
seed=0

mkdir -p ${output_dir}

base_args="${data_root} --stream ${stream} --samples_per_stage ${samples} \
  -a ${arch} -c ${cloud_arch} -b ${bs} \
  --ctx_init ${ctx_init} --tau ${tau} --beta ${beta} --lr ${lr} \
  --uplink_percent ${uplink_percent} --selection_p ${selection_p} \
  --tta_steps ${tta_steps} --seed ${seed} \
  --output_dir ${output_dir}"

echo "============================================================"
echo "  Continual TTA MVP — Student Ablation (A vs B)"
echo "  Stream: ${stream}  (${n_stages} stages)"
echo "  ${samples} samples/stage  |  uplink: ${uplink_percent}%  |  tta_steps: ${tta_steps}"
echo "  GPUs: A=${gpu0}  B=${gpu1}"
echo "  Output: ${output_dir}/"
echo "============================================================"

echo ""
echo ">>> Launching Variant A and B in parallel..."

# A (KQ_RT_RS): teacher resets, student resets each uplink
python ./ceo_continual.py ${base_args} --gpu ${gpu0} \
  --keep_queue \
  > ${output_dir}/log_A.txt 2>&1 &
pid_a=$!
echo "  A (KQ_RT_RS): --keep_queue                                  -> GPU ${gpu0}  (pid ${pid_a})"

# B (KQ_RT_KS): teacher resets, student+opt accumulate
python ./ceo_continual.py ${base_args} --gpu ${gpu1} \
  --keep_queue --keep_student --keep_student_opt \
  > ${output_dir}/log_B.txt 2>&1 &
pid_b=$!
echo "  B (KQ_RT_KS): --keep_queue --keep_student --keep_student_opt -> GPU ${gpu1}  (pid ${pid_b})"

echo ""
echo ">>> Waiting for all jobs..."

fail=0
for pid in ${pid_a} ${pid_b}; do
  wait ${pid}
  code=$?
  if [ ${code} -ne 0 ]; then
    echo "  [FAIL] pid ${pid} exited with code ${code}"
    fail=1
  fi
done

if [ ${fail} -ne 0 ]; then
  echo ""
  echo "Some jobs failed. Check logs:"
  echo "  tail ${output_dir}/log_{A,B}.txt"
  exit 1
fi
echo "  Both jobs finished successfully."

echo ""
echo "============================================================"
echo "  Done! Results in: ${output_dir}/"
echo "  - records_KQ_RT_RS.json / stream_KQ_RT_RS.png  (Variant A)"
echo "  - records_KQ_RT_KS.json / stream_KQ_RT_KS.png  (Variant B)"
echo "  - log_{A,B}.txt"
echo "============================================================"
