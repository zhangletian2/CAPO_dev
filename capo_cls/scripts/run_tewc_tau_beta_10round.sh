#!/bin/bash
# >>> capo/scripts/_paths.sh (auto-inserted)
# Locate scripts/_paths.sh relative to this script's location so the file
# stays portable after a git clone -- no absolute paths.
_CAPO_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
source "${_CAPO_SCRIPT_DIR}/../../scripts/_paths.sh"
# <<< capo/scripts/_paths.sh
# =============================================================================
# tewc τ × β fine-tuning (up50, 3-round)
#
# Fixed: keep_both_teacher_ewc, λ=10000, γ=0.95, domain-anchor
#        lr=5e-3, selection_p=0.1, tta_steps=1, uplink=50
#
# Grid:
#   tau:  {0.4, 0.5, 0.6, 0.8}
#   beta: {0.5, 0.9, 1.5}
#   + selection_p sweep {0.05, 0.2} with best tau/beta from prior run (t0.4 b0.9)
#
# 14 experiments = 4 batches (4+4+4+2)
#
# Usage:
#   bash scripts/run_tewc_tau_beta_10round.sh [gpus]
# =============================================================================

IFS=',' read -ra GPUS <<< "${1:-0,1,6,7}"
n_gpus=${#GPUS[@]}

samples=500
n_rounds=10
data_root='./dataset'
output_dir="logs/tewc_tau_beta_n${samples}_r${n_rounds}"

arch=RN50
cloud_arch=ViT-B/16
ctx_init=a_photo_of_a

ewc_lambda=10000
ewc_gamma=0.95
uplink=50

mkdir -p "${output_dir}"

base_args="${data_root} \
  --samples_per_domain ${samples} --n_rounds ${n_rounds} \
  -a ${arch} -c ${cloud_arch} -b 64 \
  --ctx_init ${ctx_init} \
  --selection_p 0.1 \
  --uplink_percent ${uplink} \
  --method keep_both_teacher_ewc \
  --ewc_lambda ${ewc_lambda} --ewc_gamma ${ewc_gamma} --ewc_anchor domain \
  --output_dir ${output_dir}"

# ── experiment grid ──────────────────────────────────────────────────────
JOBS=()

# 只保留 t0.6_b1.5, t0.7_b2.0, t0.6_b1.0 三组实验
JOBS+=("--tau 0.6 --beta 1.5|t0.6_b1.5")
JOBS+=("--tau 0.7 --beta 2.0|t0.7_b2.0")
JOBS+=("--tau 0.6 --beta 1.0|t0.6_b1.0")

total=${#JOBS[@]}

echo "============================================================"
echo "  tewc τ×β fine-tuning (up${uplink})"
echo "  T: ${cloud_arch}  S: ${arch}"
echo "  EWC: λ=${ewc_lambda} γ=${ewc_gamma} domain-anchor"
echo "  Stream: V/K x ${n_rounds} rounds, ${samples}/domain"
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
  batch_start=$((idx + 1))

  for ((slot=0; slot<n_gpus && idx<total; slot++, idx++)); do
    gpu=${GPUS[${slot}]}
    IFS='|' read -r extra tag <<< "${JOBS[${idx}]}"

    log="${output_dir}/log_${tag}.txt"
    tags+=("${tag}")

    echo ">>> [$(( idx + 1 ))/${total}] ${tag}  ->  GPU ${gpu}"
    python ./ceo_continual.py ${base_args} ${extra} --gpu ${gpu} \
      > "${log}" 2>&1 &
    pids+=($!)
  done

  echo "  Waiting for batch [${batch_start}-$(( idx ))]..."
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

# ── print summary ────────────────────────────────────────────────────────
echo "============================================================"
echo "  All ${total} jobs finished. Summary:"
echo "============================================================"
for j in "${JOBS[@]}"; do
  IFS='|' read -r _spec tag <<< "$j"
  log="${output_dir}/log_${tag}.txt"
  if [ -f "${log}" ]; then
    echo ""
    echo "--- ${tag} ---"
    grep -A1 "No-Adapt Baseline" "${log}"
    grep "S_base\|T_base" "${log}"
    grep "Gain over" "${log}"
    grep "S_V:\|S_K:" "${log}"
    grep "S_BT\|T_BT" "${log}"
  fi
done
echo ""
echo "Done. Logs in ${output_dir}/"
