#!/bin/bash
# >>> capo/scripts/_paths.sh (auto-inserted)
# Locate scripts/_paths.sh relative to this script's location so the file
# stays portable after a git clone -- no absolute paths.
_CAPO_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
source "${_CAPO_SCRIPT_DIR}/../../scripts/_paths.sh"
# <<< capo/scripts/_paths.sh
# AMS retrieval e2e benchmark (I2T + T2I).
# Usage:
#   bash scripts/run_ams_e2e_bench.sh "0,1" 500

set -e

GPU_LIST=${1:-"0,1"}
MAX_SAMPLES=${2:-500}
IFS=',' read -ra GPUS <<< "${GPU_LIST}"

if [ ${#GPUS[@]} -lt 2 ]; then
  echo "Need at least 2 GPUs, e.g. \"0,1\""
  exit 1
fi

GPU_I2T=${GPUS[0]}
GPU_T2I=${GPUS[1]}

ROOT=${RET_ROOT}
OUT_DIR=${ROOT}/results/ams_e2e_bench_ret
LOG_DIR=${ROOT}/logs/ams_e2e_bench_ret
mkdir -p "${OUT_DIR}" "${LOG_DIR}"

I2T_CSV=${OUT_DIR}/i2t.csv
T2I_CSV=${OUT_DIR}/t2i.csv

echo "========== AMS Retrieval E2E Benchmark =========="
echo "GPUs: i2t=${GPU_I2T}, t2i=${GPU_T2I}"
echo "max_samples=${MAX_SAMPLES}"
echo "================================================="

# I2T (same core hyperparams as tta_coco_ams_v3.sh)
CUDA_VISIBLE_DEVICES=${GPU_I2T} python "${ROOT}/ams_ret_e2e_bench.py" \
  --task i2t \
  --cfg-path "${ROOT}/lavis/projects/clip/exp_coco_dual_image_ret_tta.yaml" \
  --output-csv "${I2T_CSV}" \
  --output "${RET_ROOT}/output/ams_ret_e2e_i2t" \
  --arch RN50 \
  --reward_arch ViT-B-16 \
  --tta_steps 1 \
  --lr 0.0025 \
  --weight_decay 5e-4 \
  --sample_k 20 \
  --reward_process 1 \
  --process_batch 0 \
  --reward_amplify 0 \
  --multiple_reward_models 0 \
  --momentum_update 0 \
  --update_freq 64 \
  --tta_momentum 0.9998 \
  --update_w 1.0 \
  --max-samples "${MAX_SAMPLES}" \
  > "${LOG_DIR}/i2t.log" 2>&1 &

# T2I
CUDA_VISIBLE_DEVICES=${GPU_T2I} python "${ROOT}/ams_ret_e2e_bench.py" \
  --task t2i \
  --cfg-path "${ROOT}/lavis/projects/clip/exp_coco_ret_tta.yaml" \
  --output-csv "${T2I_CSV}" \
  --output "${RET_ROOT}/output/ams_ret_e2e_t2i" \
  --arch RN50 \
  --reward_arch ViT-B-16 \
  --tta_steps 1 \
  --lr 0.0025 \
  --weight_decay 5e-4 \
  --sample_k 12 \
  --reward_process 1 \
  --process_batch 0 \
  --reward_amplify 0 \
  --multiple_reward_models 0 \
  --momentum_update 0 \
  --update_freq 64 \
  --tta_momentum 0.9998 \
  --update_w 1.0 \
  --max-samples "${MAX_SAMPLES}" \
  > "${LOG_DIR}/t2i.log" 2>&1 &

wait

python "${ROOT}/gen_ams_ret_e2e_tables.py" \
  --i2t-csv "${I2T_CSV}" \
  --t2i-csv "${T2I_CSV}" \
  --out-csv "${OUT_DIR}/summary.csv" \
  > "${LOG_DIR}/summary.log" 2>&1

echo "Done."
echo "CSVs: ${OUT_DIR}"
echo "Logs: ${LOG_DIR}"
