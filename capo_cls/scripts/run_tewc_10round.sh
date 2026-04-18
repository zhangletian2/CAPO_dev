#!/bin/bash
# >>> capo/scripts/_paths.sh (auto-inserted)
# Locate scripts/_paths.sh relative to this script's location so the file
# stays portable after a git clone -- no absolute paths.
_CAPO_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
source "${_CAPO_SCRIPT_DIR}/../../scripts/_paths.sh"
# <<< capo/scripts/_paths.sh
# =============================================================================
# tewc 10-round validation: best configs from 3-round tuning
#
# Fixed: keep_both_teacher_ewc, λ=10000, γ=0.95, domain-anchor, τ=0.4
#
# Jobs:
#   1) tewc up50  β=5.0  (best V with 50% uplink)
#   2) tewc up100 β=2.0  (balanced, both > no_adapt)
#   3) tewc up100 β=5.0  (best V, positive BT)
#   4) keep_student up100 (reset teacher baseline for comparison)
#
# Usage:
#   bash scripts/run_tewc_10round.sh [gpus]
# =============================================================================

IFS=',' read -ra GPUS <<< "${1:-0,1,6,7}"
n_gpus=${#GPUS[@]}

samples=500
n_rounds=10
data_root='./dataset'
output_dir="logs/tewc_10round_n${samples}"

arch=RN50
cloud_arch=ViT-B/16
ctx_init=a_photo_of_a
tau=0.4

mkdir -p "${output_dir}"

common="${data_root} \
  --samples_per_domain ${samples} --n_rounds ${n_rounds} \
  -a ${arch} -c ${cloud_arch} -b 64 \
  --ctx_init ${ctx_init} --tau ${tau} \
  --selection_p 0.1 \
  --output_dir ${output_dir}"

ewc_args="--ewc_lambda 10000 --ewc_gamma 0.95 --ewc_anchor domain"

JOBS=()
JOBS+=("--method keep_both_teacher_ewc ${ewc_args} --uplink_percent 50  --beta 5.0|tewc_up50_b5")
JOBS+=("--method keep_both_teacher_ewc ${ewc_args} --uplink_percent 100 --beta 2.0|tewc_up100_b2")
JOBS+=("--method keep_both_teacher_ewc ${ewc_args} --uplink_percent 100 --beta 5.0|tewc_up100_b5")
JOBS+=("--method keep_student --uplink_percent 100 --beta 0.9|ks_up100")

total=${#JOBS[@]}

echo "============================================================"
echo "  tewc 10-round validation"
echo "  T: ${cloud_arch}  S: ${arch}  τ=${tau}"
echo "  Stream: V/K x ${n_rounds} rounds, ${samples}/domain"
echo "  Jobs: ${total}   GPUs: ${GPUS[*]}"
echo "  Output: ${output_dir}/"
echo "============================================================"
for j in "${JOBS[@]}"; do
  IFS='|' read -r _ tag <<< "$j"
  echo "    ${tag}"
done
echo ""

pids=()
tags=()

for ((idx=0; idx<total; idx++)); do
  slot=$((idx % n_gpus))
  gpu=${GPUS[${slot}]}
  IFS='|' read -r extra tag <<< "${JOBS[${idx}]}"
  log="${output_dir}/log_${tag}.txt"
  tags+=("${tag}")

  echo ">>> ${tag}  ->  GPU ${gpu}"
  python ./ceo_continual.py ${common} --gpu ${gpu} ${extra} \
    > "${log}" 2>&1 &
  pids+=($!)
done

echo ""
echo "All ${total} jobs launched. Waiting..."

fail=0
for i in "${!pids[@]}"; do
  wait ${pids[$i]}
  code=$?
  if [ ${code} -ne 0 ]; then
    echo "  [FAIL] ${tags[$i]}  (exit ${code})"
    fail=1
  else
    echo "  [OK]   ${tags[$i]}"
  fi
done

echo ""
if [ ${fail} -ne 0 ]; then
  echo "Some jobs failed."
  exit 1
fi

echo "============================================================"
echo "  All done. Summaries:"
echo "============================================================"
for tag in "${tags[@]}"; do
  echo ""
  echo "--- ${tag} ---"
  tail -30 "${output_dir}/log_${tag}.txt"
done
