#!/bin/bash
# >>> capo/scripts/_paths.sh (auto-inserted)
# Locate scripts/_paths.sh relative to this script's location so the file
# stays portable after a git clone -- no absolute paths.
_CAPO_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
source "${_CAPO_SCRIPT_DIR}/../../scripts/_paths.sh"
# <<< capo/scripts/_paths.sh
#
# Encode-then-decode COCO val2014 (karpathy_test split, 5000 images) at QP=1
# to produce the "cloud-side" reconstructions needed by the retrieval task.
#
# Output: ${DATA_RET}/coco2014_cloud/val2014/
#
# Only QP=1 is produced.  The original pipeline also supported QP=7 (written
# to coco2014_cloud_qp7/); that variant is intentionally NOT generated here.
#
# Usage:
#   bash capo/enc2dec/scripts/preprocess_ret.sh [N_SHARDS] [GPU_LIST]
#   bash capo/enc2dec/scripts/preprocess_ret.sh 4 "0,1,2,3"
#
# Time: ~15 min with 4 GPUs in parallel (5000 images, ~320x240 each).

set -euo pipefail

N_SHARDS=${1:-4}
GPU_LIST=${2:-"0,1,2,3"}
IFS=',' read -r -a GPUS <<< "$GPU_LIST"
if [ "${#GPUS[@]}" -lt "$N_SHARDS" ]; then
    echo "ERROR: N_SHARDS=$N_SHARDS but only ${#GPUS[@]} GPUs were given ($GPU_LIST)." >&2
    exit 1
fi

QP=1
MODEL_TYPE=0

cd "${ENC2DEC_ROOT}"
mkdir -p logs intermediate

INPUT="${DATA_RET}/coco2014/val2014"
JSON="${DATA_RET}/coco2014/coco_karpathy_test.json"
OUTPUT="${DATA_RET}/coco2014_cloud/val2014"
INTER="intermediate/ret_coco_test_qp${QP}"

mkdir -p "${OUTPUT}"

export OMP_NUM_THREADS=6
export MKL_NUM_THREADS=6
export OPENBLAS_NUM_THREADS=6

echo "================================================================"
echo ">>> ret: coco2014 val2014  (karpathy_test split, 5000 images)"
echo "    input : ${INPUT}"
echo "    output: ${OUTPUT}"
echo "    qp=${QP}, N_SHARDS=${N_SHARDS}, GPUs=${GPU_LIST}"
echo "================================================================"

for i in $(seq 0 $((N_SHARDS - 1))); do
    gpu=${GPUS[$i]}
    log="logs/ret_coco_test_qp${QP}_shard${i}.log"
    echo "  shard ${i}/${N_SHARDS} on GPU ${gpu}  -> ${log}"
    CUDA_VISIBLE_DEVICES=${gpu} python AppEncDec.py --command all \
        --input "${INPUT}" \
        --output "${OUTPUT}" \
        --json "${JSON}" \
        --intermediate "${INTER}" \
        --shard "${i}/${N_SHARDS}" \
        --qp ${QP} --model_type ${MODEL_TYPE} --device cuda:0 \
        > "${log}" 2>&1 &
done

echo "  All ${N_SHARDS} shards launched. Waiting..."
wait
echo
echo "=== ret preprocessing done (QP=${QP}) ==="
