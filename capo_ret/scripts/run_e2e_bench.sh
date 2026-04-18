#!/bin/bash
# >>> capo/scripts/_paths.sh (auto-inserted)
# Locate scripts/_paths.sh relative to this script's location so the file
# stays portable after a git clone -- no absolute paths.
_CAPO_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
source "${_CAPO_SCRIPT_DIR}/../../scripts/_paths.sh"
# <<< capo/scripts/_paths.sh
# Retrieval E2E benchmark for I2T + T2I.
# Usage:
#   bash scripts/run_e2e_bench.sh "0,1" 500

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
OUT_DIR=${ROOT}/results/e2e_bench_ret
LOG_DIR=${ROOT}/logs/e2e_bench_ret
mkdir -p "${OUT_DIR}" "${LOG_DIR}"

I2T_CSV=${OUT_DIR}/i2t.csv
T2I_CSV=${OUT_DIR}/t2i.csv

echo "========== Retrieval E2E Benchmark =========="
echo "GPUs: i2t=${GPU_I2T}, t2i=${GPU_T2I}"
echo "max_samples=${MAX_SAMPLES}"
echo "============================================="

# I2T: hyper-params from ceo_clip_coco_ret_i2t_nctx_4/hparams_image2text.json
CUDA_VISIBLE_DEVICES=${GPU_I2T} python "${ROOT}/ceo_ret_e2e_bench.py" \
  --task i2t \
  --cfg-path "${ROOT}/lavis/projects/clip/exp_coco_dual_image_ret_tta.yaml" \
  --output-csv "${I2T_CSV}" \
  --output "${RET_ROOT}/output/ceo_ret_e2e_i2t" \
  --edge RN50 \
  --cloud ViT-B-16 \
  --n_ctx 4 \
  --ctx_init "a photo of a" \
  --tta_steps 1 \
  --lr 0.0025 \
  --tau 0.8 \
  --beta 0.5 \
  --sim_top_k 256 \
  --ent_top_k 40 \
  --ent_tea 1 \
  --ent_stu 1 \
  --max-samples "${MAX_SAMPLES}" \
  > "${LOG_DIR}/i2t.log" 2>&1 &

# T2I: follow current CEO t2i setting
CUDA_VISIBLE_DEVICES=${GPU_T2I} python "${ROOT}/ceo_ret_e2e_bench.py" \
  --task t2i \
  --cfg-path "${ROOT}/lavis/projects/clip/exp_coco_ret_tta.yaml" \
  --output-csv "${T2I_CSV}" \
  --output "${RET_ROOT}/output/ceo_ret_e2e_t2i" \
  --edge RN50 \
  --cloud ViT-B-16 \
  --n_ctx 4 \
  --ctx_init "a photo of a" \
  --tta_steps 1 \
  --lr 0.0025 \
  --tau 1.0 \
  --beta 3.4 \
  --ent_top_k 40 \
  --max-samples "${MAX_SAMPLES}" \
  > "${LOG_DIR}/t2i.log" 2>&1 &

wait

python "${ROOT}/gen_ret_e2e_tables.py" \
  --i2t-csv "${I2T_CSV}" \
  --t2i-csv "${T2I_CSV}" \
  --out-csv "${OUT_DIR}/summary.csv" \
  > "${LOG_DIR}/summary.log" 2>&1

echo "Done."
echo "CSVs: ${OUT_DIR}"
echo "Logs: ${LOG_DIR}"
