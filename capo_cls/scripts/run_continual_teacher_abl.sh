#!/bin/bash
# >>> capo/scripts/_paths.sh (auto-inserted)
# Locate scripts/_paths.sh relative to this script's location so the file
# stays portable after a git clone -- no absolute paths.
_CAPO_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
source "${_CAPO_SCRIPT_DIR}/../../scripts/_paths.sh"
# <<< capo/scripts/_paths.sh
# =============================================================================
# Continual TTA MVP — Teacher Reset Ablation (C vs D)
#
# Tests: what happens when teacher does NOT reset on uplink?
# Compare with A/B (teacher resets) to isolate teacher-reset effect.
#
# C (KQ_KT_RS): teacher accumulates, student resets  → control for A
# D (KQ_KT_KS): teacher accumulates, student keeps   → control for B
#
# 2x2 factorial: {RT,KT} x {RS,KS}
#
# Usage:
#   bash scripts/run_continual_teacher_abl.sh [gpus] [samples] [stream]
#
# Examples:
#   bash scripts/run_continual_teacher_abl.sh 0,1 1000                    # default 9-stage
#   bash scripts/run_continual_teacher_abl.sh 0,1 1000 K/V/K/V/K          # custom 5-stage
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
echo "  Continual TTA MVP — Teacher Ablation (C vs D)"
echo "  Stream: ${stream}  (${n_stages} stages)"
echo "  ${samples} samples/stage  |  uplink: ${uplink_percent}%  |  tta_steps: ${tta_steps}"
echo "  GPUs: C=${gpu0}  D=${gpu1}"
echo "  Output: ${output_dir}/"
echo "============================================================"

echo ""
echo ">>> Launching Variant C and D in parallel..."

# C (KQ_KT_RS): teacher keeps, student resets
python ./ceo_continual.py ${base_args} --gpu ${gpu0} \
  --keep_queue --keep_teacher --keep_teacher_opt \
  > ${output_dir}/log_C.txt 2>&1 &
pid_c=$!
echo "  C (KQ_KT_RS): --keep_queue --keep_teacher --keep_teacher_opt                                  -> GPU ${gpu0}  (pid ${pid_c})"

# D (KQ_KT_KS): teacher keeps, student+opt keep
python ./ceo_continual.py ${base_args} --gpu ${gpu1} \
  --keep_queue --keep_teacher --keep_teacher_opt --keep_student --keep_student_opt \
  > ${output_dir}/log_D.txt 2>&1 &
pid_d=$!
echo "  D (KQ_KT_KS): --keep_queue --keep_teacher --keep_teacher_opt --keep_student --keep_student_opt -> GPU ${gpu1}  (pid ${pid_d})"

echo ""
echo ">>> Waiting for all jobs..."

fail=0
for pid in ${pid_c} ${pid_d}; do
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
  echo "  tail ${output_dir}/log_{C,D}.txt"
  exit 1
fi
echo "  Both jobs finished successfully."

echo ""
echo "============================================================"
echo "  Done! Results in: ${output_dir}/"
echo "  - records_KQ_KT_RS.json / stream_KQ_KT_RS.png  (Variant C)"
echo "  - records_KQ_KT_KS.json / stream_KQ_KT_KS.png  (Variant D)"
echo "  - log_{C,D}.txt"
echo "============================================================"
