#!/bin/bash
# >>> capo/scripts/_paths.sh (auto-inserted)
# Locate scripts/_paths.sh relative to this script's location so the file
# stays portable after a git clone -- no absolute paths.
_CAPO_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
source "${_CAPO_SCRIPT_DIR}/../../scripts/_paths.sh"
# <<< capo/scripts/_paths.sh
# =============================================================================
#  Hyperparameter sweep for keep_both_ewc + AQ
#
#  Grid:
#    ewc_gamma  : 0.9  0.95  0.99          (3)
#    ewc_lambda : 0.01 0.1  1  10  100     (5)
#    Total      : 15 runs, round-robin on 8 GPUs
#
#  Usage:
#    bash scripts/sweep_ewc_aq.sh [gpus] [samples] [rounds] [uplink%]
#
#  Examples:
#    bash scripts/sweep_ewc_aq.sh 0,1,2,3,4,5,6,7          # default
#    bash scripts/sweep_ewc_aq.sh 0,1,2,3,4,5,6,7 500 10 50
#    bash scripts/sweep_ewc_aq.sh 0 10 2 50                 # quick test
# =============================================================================

set -euo pipefail

IFS=',' read -ra GPUS <<< "${1:-0,1,2,3,4,5,6,7}"
samples=${2:-500}
n_rounds=${3:-10}
uplink=${4:-50}

data_root='./dataset'
output_dir="logs/sweep_ewc_aq_n${samples}_r${n_rounds}_up${uplink}"

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
  --method keep_both_ewc \
  --output_dir ${output_dir}"

gammas=(0.9 0.95 0.99)
lambdas=(0.01 0.1 1 10 100)

n_gpus=${#GPUS[@]}
n_jobs=$(( ${#gammas[@]} * ${#lambdas[@]} ))
gpu_of() { echo "${GPUS[$(( $1 % n_gpus ))]}"; }

echo "============================================================"
echo "  Sweep: keep_both_ewc + AQ"
echo "  ewc_gamma  = ${gammas[*]}"
echo "  ewc_lambda = ${lambdas[*]}"
echo "  ${n_jobs} jobs, ${n_rounds} rounds × ${samples} samples/domain"
echo "  GPUs: ${GPUS[*]}  (${n_gpus} available)"
echo "  Output: ${output_dir}/"
echo "============================================================"
echo ""

all_pids=()
declare -a slot_pid
for (( s=0; s<n_gpus; s++ )); do slot_pid[$s]=0; done
idx=0
fail=0

for g in "${gammas[@]}"; do
  for l in "${lambdas[@]}"; do
    tag="g${g}_l${l}"
    slot=$(( idx % n_gpus ))
    gpu="${GPUS[$slot]}"
    logf="${output_dir}/log_${tag}.txt"

    # wait for the previous job on this GPU to finish
    prev=${slot_pid[$slot]}
    if [ "${prev}" -ne 0 ]; then
      if ! wait "${prev}"; then
        echo "  [FAIL] pid ${prev}"
        fail=1
      fi
    fi

    echo "  [$(( idx+1 ))/${n_jobs}] ${tag}  ->  GPU ${gpu}"
    python ./ceo_continual.py ${base_args} \
      --gpu ${gpu} --ewc_gamma ${g} --ewc_lambda ${l} \
      > "${logf}" 2>&1 &
    slot_pid[$slot]=$!
    all_pids+=($!)
    idx=$(( idx + 1 ))
  done
done

echo ""
echo "Waiting for remaining jobs …"

for (( s=0; s<n_gpus; s++ )); do
  p=${slot_pid[$s]}
  if [ "${p}" -ne 0 ]; then
    if ! wait "${p}"; then
      echo "  [FAIL] pid ${p}"
      fail=1
    fi
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
echo "  All ${n_jobs} sweep jobs finished."
echo "  Results in: ${output_dir}/"
echo ""

# Print summary table
echo "  gamma  lambda  | mean_acc_V  mean_acc_K  Δacc_V   Δacc_K    BT"
echo "  ---------------------------------------------------------------"
for g in "${gammas[@]}"; do
  for l in "${lambdas[@]}"; do
    tag="keep_both_ewc_l${l}_g${g}_AQ"
    jf="${output_dir}/results_${tag}.json"
    if [ -f "${jf}" ]; then
      python -c "
import json, sys
d = json.load(open('${jf}'))['metrics']
print(f\"  {${g}:<6}  {${l}:<6}  | {d['mean_post_acc_V']:>8.2f}  {d['mean_post_acc_K']:>8.2f}  {d['delta_acc_V']:>+7.2f}  {d['delta_acc_K']:>+7.2f}  {d['BT']:>+6.2f}\")
"
    else
      echo "  ${g}      ${l}      |  (missing)"
    fi
  done
done
echo "============================================================"
