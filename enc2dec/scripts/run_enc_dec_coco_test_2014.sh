#!/bin/bash
# >>> capo/scripts/_paths.sh (auto-inserted)
# Locate scripts/_paths.sh relative to this script's location so the file
# stays portable after a git clone -- no absolute paths.
_CAPO_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
source "${_CAPO_SCRIPT_DIR}/../../scripts/_paths.sh"
# <<< capo/scripts/_paths.sh
set -e

export OMP_NUM_THREADS=6
export MKL_NUM_THREADS=6
export OPENBLAS_NUM_THREADS=6

INPUT=${RET_ROOT}/dataset/coco2014/val2014
JSON=${RET_ROOT}/dataset/coco2014/coco_karpathy_test.json
INTERMEDIATE=intermediate_qp7/COCO_test2014
OUTPUT=${RET_ROOT}/dataset/coco2014_cloud_qp7/val2014
QP=7
MODEL_TYPE=0
N_SHARDS=4
GPUS=(0 1 2 3)

mkdir -p "$OUTPUT" logs

for i in $(seq 0 $((N_SHARDS - 1))); do
    GPU=${GPUS[$i]}
    LOG="logs/coco_test2014_qp7_s${i}.log"
    echo "Launching shard $i/$N_SHARDS on GPU $GPU -> $LOG"
    CUDA_VISIBLE_DEVICES=$GPU python AppEncDec.py --command all \
        --input "$INPUT" \
        --json "$JSON" \
        --intermediate "$INTERMEDIATE" \
        --output "$OUTPUT" \
        --qp $QP --model_type $MODEL_TYPE \
        --device cuda:0 \
        --shard "$i/$N_SHARDS" \
        > "$LOG" 2>&1 &
done

echo "All $N_SHARDS shards launched. Monitoring..."
wait
echo "Done. All shards finished."
