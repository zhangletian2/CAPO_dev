#!/bin/bash
# >>> capo/scripts/_paths.sh (auto-inserted)
# Locate scripts/_paths.sh relative to this script's location so the file
# stays portable after a git clone -- no absolute paths.
_CAPO_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
source "${_CAPO_SCRIPT_DIR}/../../scripts/_paths.sh"
# <<< capo/scripts/_paths.sh
# =============================================================================
# tewc fine scan around t0.6_b1.5 (the current best)
#
# Grid:  τ ∈ {0.5, 0.6, 0.7}  ×  β ∈ {1.0, 1.5, 2.0}
# Skip:  combinations whose log already exists in output_dir
#
# Fixed: keep_both_teacher_ewc, λ=10000, γ=0.95, domain-anchor
#        lr=5e-3, selection_p=0.1, tta_steps=1, uplink=50
#
# Usage:
#   bash scripts/run_tewc_fine_scan.sh [gpus]
# =============================================================================

IFS=',' read -ra GPUS <<< "${1:-0,1,6,7}"
n_gpus=${#GPUS[@]}

samples=500
n_rounds=3
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

# ── build grid, skip existing ────────────────────────────────────────────
JOBS=()
skipped=0

for tau in 0.5 0.6 0.7; do
  for beta in 1.0 1.5 2.0; do
    tag="t${tau}_b${beta}"
    log="${output_dir}/log_${tag}.txt"
    if [ -f "${log}" ]; then
      echo "[SKIP] ${tag}  (${log} exists)"
      skipped=$((skipped + 1))
    else
      JOBS+=("--tau ${tau} --beta ${beta}|${tag}")
    fi
  done
done

total=${#JOBS[@]}

echo ""
echo "============================================================"
echo "  tewc fine scan around t0.6_b1.5 (up${uplink})"
echo "  T: ${cloud_arch}  S: ${arch}"
echo "  EWC: λ=${ewc_lambda} γ=${ewc_gamma} domain-anchor"
echo "  Stream: V/K x ${n_rounds} rounds, ${samples}/domain"
echo "  Skipped: ${skipped}   New jobs: ${total}   GPUs: ${GPUS[*]}"
echo "  Output: ${output_dir}/"
echo "============================================================"

if [ ${total} -eq 0 ]; then
  echo "  Nothing to run — all combinations already exist."
  exit 0
fi

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
  echo "Some jobs failed. Check logs in ${output_dir}/"
  exit 1
fi

# ── print summary ────────────────────────────────────────────────────────
echo "============================================================"
echo "  All ${total} new jobs finished. Results:"
echo "============================================================"

echo ""
printf "  %-14s  %8s  %8s  %8s  %8s  %6s\n" \
       "Config" "Base_V" "Base_K" "ΔS_V" "ΔS_K" "S_BT"
printf "  %-14s  %8s  %8s  %8s  %8s  %6s\n" \
       "--------------" "--------" "--------" "--------" "--------" "------"

for tau in 0.5 0.6 0.7; do
  for beta in 1.0 1.5 2.0; do
    tag="t${tau}_b${beta}"
    log="${output_dir}/log_${tag}.txt"
    [ -f "${log}" ] || continue
    base_v=$(grep "S_V:.*base=" "${log}" | head -1 | sed 's/.*base=\([0-9.]*\).*/\1/')
    base_k=$(grep "S_K:.*base=" "${log}" | head -1 | sed 's/.*base=\([0-9.]*\).*/\1/')
    gain_v=$(grep "S_V:.*=" "${log}" | head -1 | sed 's/.*= \([-+][0-9.]*\)/\1/')
    gain_k=$(grep "S_K:.*=" "${log}" | head -1 | sed 's/.*= \([-+][0-9.]*\)/\1/')
    sbt=$(grep "S_BT" "${log}" | sed 's/.*= \([-+][0-9.]*\)/\1/')
    printf "  %-14s  %8s  %8s  %8s  %8s  %6s\n" \
           "${tag}" "${base_v}" "${base_k}" "${gain_v}" "${gain_k}" "${sbt}"
  done
done
echo ""
echo "Done. Full logs in ${output_dir}/"
