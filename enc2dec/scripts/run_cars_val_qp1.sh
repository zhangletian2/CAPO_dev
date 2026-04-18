#!/bin/bash
# >>> capo/scripts/_paths.sh (auto-inserted)
# Locate scripts/_paths.sh relative to this script's location so the file
# stays portable after a git clone -- no absolute paths.
_CAPO_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
source "${_CAPO_SCRIPT_DIR}/../../scripts/_paths.sh"
# <<< capo/scripts/_paths.sh
set -e

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4

INPUT=${CLS_ROOT}/dataset/stanford_cars/cars_train
JSON=${CLS_ROOT}/dataset/stanford_cars/split_zhou_StanfordCars.json
JSON_KEY=val
INTERMEDIATE=intermediate/cars_val
OUTPUT=${CLS_ROOT}/dataset/stanford_cars/new/cars_train
QP=1
MODEL_TYPE=0
N_SHARDS=4
GPU=7

mkdir -p "$OUTPUT" logs

echo "stanford_cars val split: 1635 images in cars_train"
echo "qp=${QP}, 4 shards on GPU ${GPU}"

for i in $(seq 0 $((N_SHARDS - 1))); do
    LOG="logs/cars_val_qp1_s${i}.log"
    echo "Launching shard $i/$N_SHARDS on GPU $GPU -> $LOG"
    CUDA_VISIBLE_DEVICES=$GPU python AppEncDec.py --command all \
        --input "$INPUT" \
        --json "$JSON" \
        --json_key "$JSON_KEY" \
        --intermediate "$INTERMEDIATE" \
        --output "$OUTPUT" \
        --qp $QP --model_type $MODEL_TYPE \
        --device cuda:0 \
        --shard "$i/$N_SHARDS" \
        > "$LOG" 2>&1 &
done

echo "All $N_SHARDS shards launched on GPU $GPU. Monitoring..."
wait
echo "Done. All shards finished."
