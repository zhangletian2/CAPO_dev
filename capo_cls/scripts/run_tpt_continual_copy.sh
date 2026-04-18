#!/bin/bash
# >>> capo/scripts/_paths.sh (auto-inserted)
# Locate scripts/_paths.sh relative to this script's location so the file
# stays portable after a git clone -- no absolute paths.
_CAPO_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
source "${_CAPO_SCRIPT_DIR}/../../scripts/_paths.sh"
# <<< capo/scripts/_paths.sh
# =============================================================================
# Single-Model Continual Prompt Optimisation: ViT-B/16 on V/K stream
#
# Experiment grid:
#   LR:      5e-3, 1e-3
#   Methods: reset, continual, ewc (multiple λ/γ combos)
#
# Jobs are distributed round-robin across available GPUs.
#
# Usage:
#   bash scripts/run_tpt_continual.sh [gpus] [samples] [n_rounds]
#
# Examples:
#   bash scripts/run_tpt_continual.sh 0,1,2,3 500 10
#   bash scripts/run_tpt_continual.sh 0,1     500 10   # 2 GPUs
#   bash scripts/run_tpt_continual.sh 0       500 10   # 1 GPU (sequential)
# =============================================================================

IFS=',' read -ra GPUS <<< "${1:-0,1,2,3}"
samples=${2:-500}
n_rounds=${3:-10}
n_gpus=${#GPUS[@]}

data_root='./dataset'
output_dir="logs/single_model_continual_n${samples}_r${n_rounds}"

arch='ViT-B/16'
ctx_init=a_photo_of_a
selection_p=0.1
bs=64

mkdir -p "${output_dir}"

# ── experiment grid ──────────────────────────────────────────────────────
#   Format: "method lr [extra_args]|tag"

JOBS=()

for lr in 5e-3; do
  lr_tag=${lr//-/}   # 5e-3 -> 5e3, 1e-3 -> 1e3

  # EWC combos: (lambda, gamma)
  for combo in "10000,0.9"; do
    IFS=',' read -r lam gam <<< "${combo}"
    JOBS+=("ewc ${lr} --ewc_lambda ${lam} --ewc_gamma ${gam}|ewc_l${lam}_g${gam}_lr${lr_tag}")
  done
done

total=${#JOBS[@]}

echo "============================================================"
echo "  Single-Model Continual Prompt Optimisation"
echo "  Arch: ${arch}   BS: ${bs}   ctx_init: ${ctx_init}"
echo "  Stream: V/K alternating x ${n_rounds} rounds"
echo "  Samples/domain: ${samples}"
echo "  Total jobs: ${total}   GPUs: ${GPUS[*]}"
echo "  Output: ${output_dir}/"
echo "============================================================"
echo ""
echo "  Job list:"
for j in "${JOBS[@]}"; do
  IFS='|' read -r _spec tag <<< "$j"
  echo "    ${tag}"
done
echo ""

# ── launch in batches of n_gpus ──────────────────────────────────────────

fail=0
idx=0

while [ ${idx} -lt ${total} ]; do
  pids=()
  tags=()

  # launch up to n_gpus jobs
  for ((slot=0; slot<n_gpus && idx<total; slot++, idx++)); do
    gpu=${GPUS[${slot}]}
    IFS='|' read -r spec tag <<< "${JOBS[${idx}]}"
    read -r method lr extra <<< "${spec}"

    log="${output_dir}/log_${tag}.txt"
    tags+=("${tag}")

    echo ">>> [$(( idx + 1 ))/${total}] ${tag}  ->  GPU ${gpu}"
    python ./tpt_continual.py ${data_root} \
      -a "${arch}" -b ${bs} --lr ${lr} \
      --ctx_init ${ctx_init} \
      --samples_per_domain ${samples} --n_rounds ${n_rounds} \
      --selection_p ${selection_p} \
      --output_dir "${output_dir}" \
      --gpu ${gpu} --method ${method} ${extra} \
      > "${log}" 2>&1 &
    pids+=($!)
  done

  # wait for this batch
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
done

if [ ${fail} -ne 0 ]; then
  echo "Some jobs failed. Check logs in ${output_dir}/log_*.txt"
  exit 1
fi

echo "============================================================"
echo "  All ${total} jobs finished successfully."
echo "  Results: ${output_dir}/results_*.json"
echo "  Logs:    ${output_dir}/log_*.txt"
echo "============================================================"
