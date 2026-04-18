#!/bin/bash
# >>> capo/scripts/_paths.sh (auto-inserted)
# Locate scripts/_paths.sh relative to this script's location so the file
# stays portable after a git clone -- no absolute paths.
_CAPO_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
source "${_CAPO_SCRIPT_DIR}/../../scripts/_paths.sh"
# <<< capo/scripts/_paths.sh
python AppEncDec.py --command all \
--input ${MLMP_ROOT:-/data4/workspace/zlt/MLMP}/data/cityscapes/leftImg8bit \
--intermediate intermediate/cityscapes \
--output ${MLMP_ROOT:-/data4/workspace/zlt/MLMP}/data/cityscapes/leftImg8bit_new \
 --qp 1 --model_type 0 --device cuda:2