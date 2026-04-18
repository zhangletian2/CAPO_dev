#!/bin/bash
# >>> capo/scripts/_paths.sh (auto-inserted)
# Locate scripts/_paths.sh relative to this script's location so the file
# stays portable after a git clone -- no absolute paths.
_CAPO_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
source "${_CAPO_SCRIPT_DIR}/../../scripts/_paths.sh"
# <<< capo/scripts/_paths.sh
# =============================================================================
# Continual TTA with EWC: K→V→K→V→K, 5-stage, same subset per dataset
#
# Fixed config: Keep-Q / Reset-T / Keep-S + EWC
# Sweep ewc_lambda on 4 GPUs in parallel
#
# Usage:
#   bash scripts/run_continual_ewc.sh [gpus] [samples_per_stage] [uplink_percent]
#
# Examples:
#   bash scripts/run_continual_ewc.sh 0,1,2,3 5000 50.0
# =============================================================================

IFS=',' read -ra GPUS <<< "${1:-0,1,2,3}"
samples=${2:-5000}
uplink_percent=${3:-50.0}

gpu0=${GPUS[0]:-0}
gpu1=${GPUS[1]:-${gpu0}}
gpu2=${GPUS[2]:-${gpu0}}
gpu3=${GPUS[3]:-${gpu0}}

data_root='./dataset'
stream='K/V/K/V/K'
output_dir="logs/continual_ewc_KVKVK_n${samples}_up${uplink_percent}"

arch=RN50
cloud_arch=ViT-B/16
ctx_init=a_photo_of_a
selection_p=0.1
tau=0.4
beta=0.9
bs=64

mkdir -p ${output_dir}

base_args="${data_root} --stream ${stream} --samples_per_stage ${samples} \
  -a ${arch} -c ${cloud_arch} -b ${bs} \
  --ctx_init ${ctx_init} --tau ${tau} --beta ${beta} \
  --uplink_percent ${uplink_percent} --selection_p ${selection_p} \
  --output_dir ${output_dir}"

lambdas=(0.0 1.0 10.0 100.0)

echo "============================================================"
echo "  Continual TTA + EWC  (4 GPUs parallel)"
echo "  Stream: ${stream}  |  ${samples} samples/stage"
echo "  uplink_percent: ${uplink_percent}%"
echo "  ewc_lambda: ${lambdas[*]}"
echo "  GPUs: ${gpu0}, ${gpu1}, ${gpu2}, ${gpu3}"
echo "  Output: ${output_dir}/"
echo "============================================================"

echo ""
echo ">>> Launching 4 lambda settings in parallel..."

python ./ceo_continual.py ${base_args} --gpu ${gpu0} --ewc_lambda ${lambdas[0]} \
  > ${output_dir}/log_lam${lambdas[0]}.txt 2>&1 &
pid0=$!
echo "  λ=${lambdas[0]}  -> GPU ${gpu0}  (pid ${pid0})"

python ./ceo_continual.py ${base_args} --gpu ${gpu1} --ewc_lambda ${lambdas[1]} \
  > ${output_dir}/log_lam${lambdas[1]}.txt 2>&1 &
pid1=$!
echo "  λ=${lambdas[1]}  -> GPU ${gpu1}  (pid ${pid1})"

python ./ceo_continual.py ${base_args} --gpu ${gpu2} --ewc_lambda ${lambdas[2]} \
  > ${output_dir}/log_lam${lambdas[2]}.txt 2>&1 &
pid2=$!
echo "  λ=${lambdas[2]}  -> GPU ${gpu2}  (pid ${pid2})"

python ./ceo_continual.py ${base_args} --gpu ${gpu3} --ewc_lambda ${lambdas[3]} \
  > ${output_dir}/log_lam${lambdas[3]}.txt 2>&1 &
pid3=$!
echo "  λ=${lambdas[3]}  -> GPU ${gpu3}  (pid ${pid3})"

echo ""
echo ">>> Waiting for all jobs..."

fail=0
for pid in ${pid0} ${pid1} ${pid2} ${pid3}; do
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
  for lam in "${lambdas[@]}"; do
    echo "  tail ${output_dir}/log_lam${lam}.txt"
  done
  exit 1
fi
echo "  All 4 jobs finished successfully."

echo ""
echo "============================================================"
echo "  Done! Results in: ${output_dir}/"
echo "  - records_EWC_lam*.json    (per-sample records)"
echo "  - stream_EWC_lam*.png      (individual curves)"
echo "  - log_lam*.txt             (per-job stdout/stderr)"
echo "============================================================"
