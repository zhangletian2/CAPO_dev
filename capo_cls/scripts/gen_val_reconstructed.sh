#!/bin/bash
# >>> capo/scripts/_paths.sh (auto-inserted)
# Locate scripts/_paths.sh relative to this script's location so the file
# stays portable after a git clone -- no absolute paths.
_CAPO_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
source "${_CAPO_SCRIPT_DIR}/../../scripts/_paths.sh"
# <<< capo/scripts/_paths.sh
# Generate reconstructed (new/) images for ALL splits (train/val/test) of each fine-grained dataset.
# Only needed once. Already existing files in new/ will be overwritten (same content).
# Aircraft already has all images reconstructed, so it's skipped.
#
# Usage: bash gen_val_reconstructed.sh <device>
# Example: bash gen_val_reconstructed.sh cuda:0

DEVICE=${1:-cuda:0}
QP=1
MODEL_TYPE=0
ENC2DEC_DIR=${ENC2DEC_ROOT}
DATASET_ROOT=$(realpath ./dataset)

cd ${ENC2DEC_DIR}

declare -A DS_INPUT
DS_INPUT[flower102]="${DATASET_ROOT}/oxford_flowers/jpg"
DS_INPUT[dtd]="${DATASET_ROOT}/dtd/images"
DS_INPUT[pets]="${DATASET_ROOT}/oxford_pets/images"
DS_INPUT[cars]="${DATASET_ROOT}/stanford_cars"
DS_INPUT[ucf101]="${DATASET_ROOT}/ucf101/UCF-101-midframes"
DS_INPUT[caltech101]="${DATASET_ROOT}/caltech-101/101_ObjectCategories"
DS_INPUT[food101]="${DATASET_ROOT}/food-101/images"
DS_INPUT[sun397]="${DATASET_ROOT}/sun397/SUN397"
DS_INPUT[eurosat]="${DATASET_ROOT}/eurosat/2750"

declare -A DS_OUTPUT
DS_OUTPUT[flower102]="${DATASET_ROOT}/oxford_flowers/new"
DS_OUTPUT[dtd]="${DATASET_ROOT}/dtd/new"
DS_OUTPUT[pets]="${DATASET_ROOT}/oxford_pets/new"
DS_OUTPUT[cars]="${DATASET_ROOT}/stanford_cars/new"
DS_OUTPUT[ucf101]="${DATASET_ROOT}/ucf101/new"
DS_OUTPUT[caltech101]="${DATASET_ROOT}/caltech-101/new"
DS_OUTPUT[food101]="${DATASET_ROOT}/food-101/new"
DS_OUTPUT[sun397]="${DATASET_ROOT}/sun397/new"
DS_OUTPUT[eurosat]="${DATASET_ROOT}/eurosat/new"

for ds in flower102 dtd pets cars ucf101 caltech101 food101 sun397 eurosat; do
    INPUT=${DS_INPUT[$ds]}
    OUTPUT=${DS_OUTPUT[$ds]}
    INTER=intermediate/val_${ds}

    echo "============================================"
    echo ">>> Reconstructing: ${ds}"
    echo "    Input:  ${INPUT}"
    echo "    Output: ${OUTPUT}"
    echo "============================================"

    python AppEncDec.py --command all \
        --input ${INPUT} \
        --intermediate ${INTER} \
        --output ${OUTPUT} \
        --qp ${QP} --model_type ${MODEL_TYPE} --device ${DEVICE}

    echo ">>> Done: ${ds}"
    echo ""
done

echo "=== All datasets reconstructed ==="
