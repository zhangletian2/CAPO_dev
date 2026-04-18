#!/bin/bash
# >>> capo/scripts/_paths.sh (auto-inserted)
# Locate scripts/_paths.sh relative to this script's location so the file
# stays portable after a git clone -- no absolute paths.
_CAPO_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
source "${_CAPO_SCRIPT_DIR}/../../scripts/_paths.sh"
# <<< capo/scripts/_paths.sh
# =============================================================================
# Fullscale 5-Domain continual adaptation (I→A→V→R→K, all samples)
#
# 4 configs:
#   reset          CEO default baseline
#   best           tewc_aq l=5000 g=0.85 t=0.8 b=2.0 lr=2.5e-3 sp=0.05
#   conservative   tewc_aq l=10000 g=0.85 t=0.8 b=2.0 lr=1e-3 sp=0.05
#   sp0.1          tewc_aq l=5000 g=0.85 t=0.8 b=2.0 lr=2.5e-3 sp=0.1
#
# Usage:
#   bash scripts/run_5d_fullscale.sh [gpus]
#   bash scripts/run_5d_fullscale.sh 1,2,3,4
# =============================================================================

IFS=',' read -ra GPUS <<< "${1:-1,2,3,4}"
n_gpus=${#GPUS[@]}

data_root='./dataset'
output_dir="logs/5d_fullscale"

mkdir -p "${output_dir}"

common="${data_root} \
  --domains IAVRK \
  --samples_per_domain 999999 \
  -a RN50 -c ViT-B/16 -b 64 \
  --ctx_init a_photo_of_a \
  --uplink_percent 50 \
  --ewc_anchor domain \
  --output_dir ${output_dir}"

JOBS=()
JOBS+=("--method reset --tau 0.4 --beta 0.9 --lr 5e-3 --selection_p 0.1|reset")
JOBS+=("--method tewc_aq --ewc_lambda 5000 --ewc_gamma 0.85 --tau 0.8 --beta 2.0 --lr 2.5e-3 --selection_p 0.05|best")
JOBS+=("--method tewc_aq --ewc_lambda 10000 --ewc_gamma 0.85 --tau 0.8 --beta 2.0 --lr 1e-3 --selection_p 0.05|conservative")
JOBS+=("--method tewc_aq --ewc_lambda 5000 --ewc_gamma 0.85 --tau 0.8 --beta 2.0 --lr 2.5e-3 --selection_p 0.1|sp0.1")

total=${#JOBS[@]}

echo "============================================================"
echo "  Fullscale 5-Domain: I(50k) A(7.5k) V(10k) R(30k) K(51k)"
echo "  Total: ${total} runs   GPUs: ${GPUS[*]}"
echo "  Output: ${output_dir}/"
echo "============================================================"
for j in "${JOBS[@]}"; do
  IFS='|' read -r _ tag <<< "$j"
  echo "    ${tag}"
done
echo ""

pids=()
tags=()
for ((i=0; i<total && i<n_gpus; i++)); do
  gpu=${GPUS[$i]}
  IFS='|' read -r extra tag <<< "${JOBS[$i]}"
  log="${output_dir}/log_${tag}.txt"
  tags+=("${tag}")
  echo ">>> [$(( i+1 ))/${total}] ${tag} -> GPU ${gpu}"
  python ./ceo_continual_5d.py ${common} ${extra} --gpu ${gpu} \
    > "${log}" 2>&1 &
  pids+=($!)
done

echo "  Waiting (~15-20h)..."
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

if [ ${fail} -ne 0 ]; then
  echo "Some jobs failed. Check ${output_dir}/log_*.txt"
  exit 1
fi

echo ""
echo "============================================================"
echo "  Fullscale Summary"
echo "============================================================"
printf "  %-20s  %8s  %8s  %8s  %8s  %6s\n" "Tag" "Baseline" "Phase1" "Phase2" "P2Gain" "BT"
echo "  -----------------------------------------------------------------------"
for j in "${JOBS[@]}"; do
  IFS='|' read -r _ tag <<< "$j"
  log="${output_dir}/log_${tag}.txt"
  if [ -f "${log}" ]; then
    bl=$(grep "Baseline.*:" -A20 "${log}" | grep "Mean" | head -1 | awk '{print $NF}')
    p1=$(grep "Phase 1 (Adaptation):" -A20 "${log}" | grep "Mean" | head -1 | awk '{print $NF}')
    p2=$(grep "Phase 2 (Frozen):" -A20 "${log}" | grep "Mean" | head -1 | awk '{print $NF}')
    p2g=$(grep "Phase2 Gain.*:" -A10 "${log}" | grep "Mean:" | awk '{print $NF}')
    bt=$(grep "BT = " "${log}" | awk '{print $NF}')
    printf "  %-20s  %8s  %8s  %8s  %8s  %6s\n" "${tag}" "${bl}" "${p1}" "${p2}" "${p2g}" "${bt}"
  fi
done
echo ""
echo "Done. Logs: ${output_dir}/"
