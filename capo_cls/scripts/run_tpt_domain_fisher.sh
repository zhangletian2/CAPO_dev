#!/bin/bash
# >>> capo/scripts/_paths.sh (auto-inserted)
# Locate scripts/_paths.sh relative to this script's location so the file
# stays portable after a git clone -- no absolute paths.
_CAPO_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
source "${_CAPO_SCRIPT_DIR}/../../scripts/_paths.sh"
# <<< capo/scripts/_paths.sh
# =============================================================================
# Domain-anchor Online EWC — Fisher computed on frozen θ*_d at domain end
#
# Key idea:
#   1. Within domain d: only update θ via L_TTA + L_EWC(Ω̄_{d-1}, θ*_{d-1})
#   2. At domain end: freeze θ*_d = θ, re-iterate loader to compute
#      Ω_d = (1/N) Σ (∂L/∂θ)²  on the frozen θ*_d
#   3. Online merge: Ω̄_d = γ·Ω̄_{d-1} + Ω_d
#
# Grid:
#   lr:     5e-3
#   anchor: domain
#   λ:      100, 1000, 10000
#   γ:      0.95, 0.99
#   + baselines: reset, continual
#
# Usage:
#   bash scripts/run_tpt_domain_fisher.sh [gpus] [samples] [n_rounds]
#
# Examples:
#   bash scripts/run_tpt_domain_fisher.sh 0,1,2,3 500 5
#   bash scripts/run_tpt_domain_fisher.sh 0,1,2,3,4,5,6,7 500 5
# =============================================================================

IFS=',' read -ra GPUS <<< "${1:-0,1,2,3}"
samples=${2:-500}
n_rounds=${3:-5}
n_gpus=${#GPUS[@]}

data_root='./dataset'
output_dir="logs/domain_fisher_n${samples}_r${n_rounds}"

arch='ViT-B/16'
ctx_init=a_photo_of_a
selection_p=0.1
bs=64
lr=5e-3

mkdir -p "${output_dir}"

# ── experiment grid ──────────────────────────────────────────────────────
JOBS=()

# baselines
JOBS+=("reset ${lr}|reset")
JOBS+=("continual ${lr}|continual")

# domain-anchor EWC (Fisher on frozen θ*_d)
for combo in "100,0.95" "100,0.99" "1000,0.95" "1000,0.99" "10000,0.95" "10000,0.99"; do
  IFS=',' read -r lam gam <<< "${combo}"
  JOBS+=("ewc ${lr} --ewc_lambda ${lam} --ewc_gamma ${gam} --ewc_anchor domain|da_l${lam}_g${gam}")
done

total=${#JOBS[@]}

echo "============================================================"
echo "  Domain-Anchor Online EWC (Fisher on frozen θ*_d)"
echo "  Arch: ${arch}   lr: ${lr}   ctx_init: ${ctx_init}"
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

# ── launch in batches ────────────────────────────────────────────────────

fail=0
idx=0

while [ ${idx} -lt ${total} ]; do
  pids=()
  tags=()

  for ((slot=0; slot<n_gpus && idx<total; slot++, idx++)); do
    gpu=${GPUS[${slot}]}
    IFS='|' read -r spec tag <<< "${JOBS[${idx}]}"
    read -r method lr_val extra <<< "${spec}"

    log="${output_dir}/log_${tag}.txt"
    tags+=("${tag}")

    echo ">>> [$(( idx + 1 ))/${total}] ${tag}  ->  GPU ${gpu}"
    python ./tpt_continual.py ${data_root} \
      -a "${arch}" -b ${bs} --lr ${lr_val} \
      --ctx_init ${ctx_init} \
      --samples_per_domain ${samples} --n_rounds ${n_rounds} \
      --selection_p ${selection_p} \
      --output_dir "${output_dir}" \
      --gpu ${gpu} --method ${method} ${extra} \
      > "${log}" 2>&1 &
    pids+=($!)
  done

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
