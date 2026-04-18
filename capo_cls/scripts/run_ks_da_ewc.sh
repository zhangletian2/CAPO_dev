#!/bin/bash
# >>> capo/scripts/_paths.sh (auto-inserted)
# Locate scripts/_paths.sh relative to this script's location so the file
# stays portable after a git clone -- no absolute paths.
_CAPO_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
source "${_CAPO_SCRIPT_DIR}/../../scripts/_paths.sh"
# <<< capo/scripts/_paths.sh
# =============================================================================
# keep_student + domain-anchor EWC on student
#
# Config: teacher resets every sample, student keeps + domain-anchor EWC
# Teacher: reset prompt + optimizer each sample, loss = ent (no EWC)
# Student: keep prompt, loss = ent + kl + ewc (domain-anchor)
#
# Baselines (reset / keep_student / keep_both) already in:
#   logs/continual_vk_n500_r10_up50/
#
# Grid:
#   λ:  1000, 10000
#   γ:  0.9, 0.95
#   anchor: domain
#   queue: accumulate (keep)
#
# Usage:
#   bash scripts/run_ks_da_ewc.sh [gpus] [samples] [n_rounds] [uplink]
# =============================================================================

IFS=',' read -ra GPUS <<< "${1:-0,1,2,3}"
samples=${2:-500}
n_rounds=${3:-10}
uplink=${4:-50}
n_gpus=${#GPUS[@]}

data_root='./dataset'
output_dir="logs/ks_da_ewc_n${samples}_r${n_rounds}_up${uplink}"

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

# ── experiment grid ──────────────────────────────────────────────────────
JOBS=()

for combo in "100,0.9" "100,0.95" "1000,0.9" "1000,0.95" "10000,0.9" "10000,0.95"; do
  IFS=',' read -r lam gam <<< "${combo}"
  JOBS+=("--method keep_student_ewc --ewc_lambda ${lam} --ewc_gamma ${gam} --ewc_anchor domain|ks_da_l${lam}_g${gam}")
done

total=${#JOBS[@]}

echo "============================================================"
echo "  keep_student + domain-anchor EWC (student only)"
echo "  T: ${cloud_arch} (reset)  S: ${arch} (keep+EWC)"
echo "  Stream: V/K x ${n_rounds} rounds, ${samples}/domain"
echo "  Queue: accumulate (keep)   Uplink: ${uplink}%"
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

# ── launch ───────────────────────────────────────────────────────────────

fail=0
idx=0

while [ ${idx} -lt ${total} ]; do
  pids=()
  tags=()

  for ((slot=0; slot<n_gpus && idx<total; slot++, idx++)); do
    gpu=${GPUS[${slot}]}
    IFS='|' read -r extra tag <<< "${JOBS[${idx}]}"

    log="${output_dir}/log_${tag}.txt"
    tags+=("${tag}")

    echo ">>> [$(( idx + 1 ))/${total}] ${tag}  ->  GPU ${gpu}"
    python ./ceo_continual.py ${base_args} --gpu ${gpu} ${extra} \
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
