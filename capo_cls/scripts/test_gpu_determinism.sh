#!/bin/bash
# >>> capo/scripts/_paths.sh (auto-inserted)
# Locate scripts/_paths.sh relative to this script's location so the file
# stays portable after a git clone -- no absolute paths.
_CAPO_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
source "${_CAPO_SCRIPT_DIR}/../../scripts/_paths.sh"
# <<< capo/scripts/_paths.sh
# =============================================================================
# Test GPU non-determinism & benchmark speed impact
#
# Test 1: same config, 4 GPUs, benchmark=True  → expect ~3-5% variation
# Test 2: same config, 4 GPUs, deterministic   → expect identical results
# Test 3: single GPU, benchmark True vs False   → compare speed
#
# Uses no_adapt (no TTA, pure inference) for fastest turnaround
# + one reset run (with TTA) to measure speed impact
# =============================================================================

IFS=',' read -ra GPUS <<< "${1:-0,1,6,7}"
n_gpus=${#GPUS[@]}

samples=200
n_rounds=1
data_root='./dataset'
output_dir="logs/gpu_determinism_test"

arch=RN50
cloud_arch=ViT-B/16
ctx_init=a_photo_of_a

mkdir -p "${output_dir}"

common="${data_root} \
  --samples_per_domain ${samples} --n_rounds ${n_rounds} \
  -a ${arch} -c ${cloud_arch} -b 64 \
  --ctx_init ${ctx_init} --tau 0.4 --beta 0.9 \
  --selection_p 0.1 --uplink_percent 100 \
  --no_adapt \
  --output_dir ${output_dir}"

echo "============================================================"
echo "  Test 1: no_adapt on ${n_gpus} GPUs, benchmark=True"
echo "============================================================"
pids=(); tags=()
for gpu in "${GPUS[@]}"; do
  tag="noadapt_bench_gpu${gpu}"
  tags+=("${tag}")
  python ./ceo_continual.py ${common} --gpu ${gpu} \
    > "${output_dir}/log_${tag}.txt" 2>&1 &
  pids+=($!)
  echo "  GPU ${gpu} started"
done
for i in "${!pids[@]}"; do wait ${pids[$i]}; done
echo "  Done. Results:"
for tag in "${tags[@]}"; do
  line=$(grep "S_acc'_V(1)" "${output_dir}/log_${tag}.txt")
  echo "    ${tag}: ${line}"
done

echo ""
echo "============================================================"
echo "  Test 2: no_adapt on ${n_gpus} GPUs, --deterministic"
echo "============================================================"
pids=(); tags=()
for gpu in "${GPUS[@]}"; do
  tag="noadapt_det_gpu${gpu}"
  tags+=("${tag}")
  python ./ceo_continual.py ${common} --gpu ${gpu} --deterministic \
    > "${output_dir}/log_${tag}.txt" 2>&1 &
  pids+=($!)
  echo "  GPU ${gpu} started"
done
for i in "${!pids[@]}"; do wait ${pids[$i]}; done
echo "  Done. Results:"
for tag in "${tags[@]}"; do
  line=$(grep "S_acc'_V(1)" "${output_dir}/log_${tag}.txt")
  echo "    ${tag}: ${line}"
done

echo ""
echo "============================================================"
echo "  Test 3: speed benchmark (reset, 200 samples, 1 round)"
echo "          GPU ${GPUS[0]}: benchmark=True vs deterministic"
echo "============================================================"

reset_common="${data_root} \
  --samples_per_domain ${samples} --n_rounds ${n_rounds} \
  -a ${arch} -c ${cloud_arch} -b 64 \
  --ctx_init ${ctx_init} --tau 0.4 --beta 0.9 \
  --selection_p 0.1 --uplink_percent 100 \
  --method reset \
  --output_dir ${output_dir} \
  --gpu ${GPUS[0]}"

echo "  Running reset + benchmark=True ..."
python ./ceo_continual.py ${reset_common} \
  > "${output_dir}/log_speed_bench.txt" 2>&1
time_bench=$(grep -oP '\(\K[0-9]+(?=s\))' "${output_dir}/log_speed_bench.txt" | tail -1)
echo "  benchmark=True:  ${time_bench}s"

echo "  Running reset + deterministic ..."
python ./ceo_continual.py ${reset_common} --deterministic \
  > "${output_dir}/log_speed_det.txt" 2>&1
time_det=$(grep -oP '\(\K[0-9]+(?=s\))' "${output_dir}/log_speed_det.txt" | tail -1)
echo "  deterministic:   ${time_det}s"

echo ""
echo "============================================================"
echo "  Summary"
echo "============================================================"
echo "  Test 1 (benchmark, multi-GPU):"
for f in "${output_dir}"/log_noadapt_bench_gpu*.txt; do
  tag=$(basename "$f" .txt)
  sv=$(grep "S_acc'_V(1)" "$f" | grep -oP 'S_acc._V\(1\) = \K[0-9.]+')
  tv=$(grep "T_acc'_V(1)" "$f" | grep -oP 'T_acc._V\(1\) = \K[0-9.]+')
  echo "    ${tag}:  S_V=${sv}%  T_V=${tv}%"
done
echo ""
echo "  Test 2 (deterministic, multi-GPU):"
for f in "${output_dir}"/log_noadapt_det_gpu*.txt; do
  tag=$(basename "$f" .txt)
  sv=$(grep "S_acc'_V(1)" "$f" | grep -oP 'S_acc._V\(1\) = \K[0-9.]+')
  tv=$(grep "T_acc'_V(1)" "$f" | grep -oP 'T_acc._V\(1\) = \K[0-9.]+')
  echo "    ${tag}:  S_V=${sv}%  T_V=${tv}%"
done
echo ""
echo "  Test 3 (speed, GPU ${GPUS[0]}):"
echo "    benchmark=True:  ${time_bench}s"
echo "    deterministic:   ${time_det}s"
if [ -n "${time_bench}" ] && [ -n "${time_det}" ]; then
  slowdown=$(echo "scale=1; (${time_det} - ${time_bench}) * 100 / ${time_bench}" | bc 2>/dev/null)
  echo "    slowdown: ~${slowdown}%"
fi
echo "============================================================"
