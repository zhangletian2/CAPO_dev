#!/bin/bash
# >>> capo/scripts/_paths.sh (auto-inserted)
# Locate scripts/_paths.sh relative to this script's location so the file
# stays portable after a git clone -- no absolute paths.
_CAPO_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
source "${_CAPO_SCRIPT_DIR}/../../scripts/_paths.sh"
# <<< capo/scripts/_paths.sh
#
# Encode-then-decode every cls dataset at QP=1 to produce the "cloud-side"
# reconstructed images that CEO needs for cloud-edge experiments.
#
# Layout produced (for each dataset $DS):
#   ${DATA_CLS}/<DS>/new/                  <-- QP=1 reconstructions
#
# Coverage:
#   * 9 fine-grained datasets (Caltech101, DTD, EuroSAT, Food101, OxfordFlowers,
#     OxfordPets, StanfordCars, SUN397, UCF101) filtered by their split_zhou
#     json's  "test" split.
#   * fgvc_aircraft filtered by images_variant_test.txt.
#   * 4 ImageNet OOD variants (A / R / Sketch / V2) -- full sets, no filter.
#
# Only QP=1 is produced.  QP=7 variants (new_qp7/, *_cloud_qp7/) are intentionally
# NOT generated; drop into the original pipeline if you need them.
#
# Usage:
#   bash capo/enc2dec/scripts/preprocess_cls.sh [DEVICE]
#   bash capo/enc2dec/scripts/preprocess_cls.sh cuda:0
#
# Time:  ~2h on a single RTX 4090 for all 14 datasets at QP=1.

set -euo pipefail

DEVICE=${1:-cuda:0}
QP=1
MODEL_TYPE=0

cd "${ENC2DEC_ROOT}"
mkdir -p logs intermediate

CLS_DATA="${DATA_CLS}"

run_with_json() {
    local ds=$1 inp=$2 out=$3 json=$4
    echo "============================================"
    echo ">>> ${ds}  (split_zhou_*.json, key=test)"
    echo "    input : ${inp}"
    echo "    output: ${out}"
    echo "============================================"
    mkdir -p "${out}"
    python AppEncDec.py --command all \
        --input "${inp}" \
        --output "${out}" \
        --json "${json}" --json_key test \
        --intermediate "intermediate/cls_${ds}_test_qp${QP}" \
        --qp ${QP} --model_type ${MODEL_TYPE} --device "${DEVICE}"
}

run_with_txt() {
    local ds=$1 inp=$2 out=$3 txt=$4
    echo "============================================"
    echo ">>> ${ds}  (txt filter = ${txt##*/})"
    echo "    input : ${inp}"
    echo "    output: ${out}"
    echo "============================================"
    mkdir -p "${out}"
    python AppEncDec.py --command all \
        --input "${inp}" \
        --output "${out}" \
        --txt_filter "${txt}" \
        --intermediate "intermediate/cls_${ds}_test_qp${QP}" \
        --qp ${QP} --model_type ${MODEL_TYPE} --device "${DEVICE}"
}

run_all() {
    local ds=$1 inp=$2 out=$3
    echo "============================================"
    echo ">>> ${ds}  (no split filter, full set)"
    echo "    input : ${inp}"
    echo "    output: ${out}"
    echo "============================================"
    mkdir -p "${out}"
    python AppEncDec.py --command all \
        --input "${inp}" \
        --output "${out}" \
        --intermediate "intermediate/cls_${ds}_full_qp${QP}" \
        --qp ${QP} --model_type ${MODEL_TYPE} --device "${DEVICE}"
}

# ------------- 9 fine-grained datasets (split_zhou JSON, test split) -------------
run_with_json oxford_flowers \
    "${CLS_DATA}/oxford_flowers/jpg" \
    "${CLS_DATA}/oxford_flowers/new" \
    "${CLS_DATA}/oxford_flowers/split_zhou_OxfordFlowers.json"

run_with_json dtd \
    "${CLS_DATA}/dtd/images" \
    "${CLS_DATA}/dtd/new" \
    "${CLS_DATA}/dtd/split_zhou_DescribableTextures.json"

run_with_json oxford_pets \
    "${CLS_DATA}/oxford_pets/images" \
    "${CLS_DATA}/oxford_pets/new" \
    "${CLS_DATA}/oxford_pets/split_zhou_OxfordPets.json"

run_with_json stanford_cars \
    "${CLS_DATA}/stanford_cars" \
    "${CLS_DATA}/stanford_cars/new" \
    "${CLS_DATA}/stanford_cars/split_zhou_StanfordCars.json"

run_with_json ucf101 \
    "${CLS_DATA}/ucf101/UCF-101-midframes" \
    "${CLS_DATA}/ucf101/new" \
    "${CLS_DATA}/ucf101/split_zhou_UCF101.json"

run_with_json caltech101 \
    "${CLS_DATA}/caltech-101/101_ObjectCategories" \
    "${CLS_DATA}/caltech-101/new" \
    "${CLS_DATA}/caltech-101/split_zhou_Caltech101.json"

run_with_json food101 \
    "${CLS_DATA}/food-101/images" \
    "${CLS_DATA}/food-101/new" \
    "${CLS_DATA}/food-101/split_zhou_Food101.json"

run_with_json sun397 \
    "${CLS_DATA}/sun397/SUN397" \
    "${CLS_DATA}/sun397/new" \
    "${CLS_DATA}/sun397/split_zhou_SUN397.json"

run_with_json eurosat \
    "${CLS_DATA}/eurosat/2750" \
    "${CLS_DATA}/eurosat/new" \
    "${CLS_DATA}/eurosat/split_zhou_EuroSAT.json"

# ------------- fgvc_aircraft (txt filter, test split) -------------
run_with_txt fgvc_aircraft \
    "${CLS_DATA}/fgvc_aircraft/images" \
    "${CLS_DATA}/fgvc_aircraft/new" \
    "${CLS_DATA}/fgvc_aircraft/images_variant_test.txt"

# ------------- 4 ImageNet OOD variants (no split filter, full set) -------------
run_all imagenet-adversarial \
    "${CLS_DATA}/imagenet-adversarial/imagenet-a" \
    "${CLS_DATA}/imagenet-adversarial/new"

run_all imagenet-rendition \
    "${CLS_DATA}/imagenet-rendition/imagenet-r" \
    "${CLS_DATA}/imagenet-rendition/new"

run_all imagenet-sketch \
    "${CLS_DATA}/imagenet-sketch/images" \
    "${CLS_DATA}/imagenet-sketch/new"

run_all imagenetv2 \
    "${CLS_DATA}/imagenetv2/imagenetv2-matched-frequency-format-val" \
    "${CLS_DATA}/imagenetv2/new"

echo
echo "=== cls preprocessing done (QP=${QP}, device=${DEVICE}) ==="
