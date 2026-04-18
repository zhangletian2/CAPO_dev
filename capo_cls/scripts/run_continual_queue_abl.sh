#!/bin/bash
# >>> capo/scripts/_paths.sh (auto-inserted)
# Locate scripts/_paths.sh relative to this script's location so the file
# stays portable after a git clone -- no absolute paths.
_CAPO_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
source "${_CAPO_SCRIPT_DIR}/../../scripts/_paths.sh"
# <<< capo/scripts/_paths.sh
# =============================================================================
# Continual TTA MVP — Queue Ablation: Reset-Q + Reset-T + {Reset-S, Keep-S}
#
# Isolates the effect of entropy queue persistence by comparing with A/B:
#   E (RQ_RT_RS) vs A (KQ_RT_RS) → queue-reset effect (student=reset)
#   F (RQ_RT_KS) vs B (KQ_RT_KS) → queue-reset effect (student=keep)
#
# E: entropy queue cleared at every stage boundary, teacher & student reset
# F: entropy queue cleared at every stage boundary, teacher resets, student keeps
#
# Usage:
#   bash scripts/run_continual_queue_abl.sh [gpus] [samples] [stream]
#
# Examples:
#   bash scripts/run_continual_queue_abl.sh 0,1 1000
#   bash scripts/run_continual_queue_abl.sh 0,1 1000 K/V/K/V/K
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
echo "  Continual TTA MVP — Queue Ablation (E vs F)"
echo "  Stream: ${stream}  (${n_stages} stages)"
echo "  ${samples} samples/stage  |  uplink: ${uplink_percent}%  |  tta_steps: ${tta_steps}"
echo "  GPUs: E=${gpu0}  F=${gpu1}"
echo "  Output: ${output_dir}/"
echo "============================================================"

echo ""
echo ">>> Launching Variant E and F in parallel..."

# E (RQ_RT_RS): queue resets at boundary, teacher resets, student resets
python ./ceo_continual.py ${base_args} --gpu ${gpu0} \
  > ${output_dir}/log_E.txt 2>&1 &
pid_e=$!
echo "  E (RQ_RT_RS): (no --keep_queue)                             -> GPU ${gpu0}  (pid ${pid_e})"

# F (RQ_RT_KS): queue resets at boundary, teacher resets, student+opt keep
python ./ceo_continual.py ${base_args} --gpu ${gpu1} \
  --keep_student --keep_student_opt \
  > ${output_dir}/log_F.txt 2>&1 &
pid_f=$!
echo "  F (RQ_RT_KS): --keep_student --keep_student_opt             -> GPU ${gpu1}  (pid ${pid_f})"

echo ""
echo ">>> Waiting for all jobs..."

fail=0
for pid in ${pid_e} ${pid_f}; do
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
  echo "  tail ${output_dir}/log_{E,F}.txt"
  exit 1
fi
echo "  Both jobs finished successfully."

echo ""
echo "============================================================"
echo "  Done! Results in: ${output_dir}/"
echo "  - records_RQ_RT_RS.json / stream_RQ_RT_RS.png  (Variant E)"
echo "  - records_RQ_RT_KS.json / stream_RQ_RT_KS.png  (Variant F)"
echo "  - log_{E,F}.txt"
echo "============================================================"
