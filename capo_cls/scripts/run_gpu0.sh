#!/bin/bash
# >>> capo/scripts/_paths.sh (auto-inserted)
# Locate scripts/_paths.sh relative to this script's location so the file
# stays portable after a git clone -- no absolute paths.
_CAPO_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
source "${_CAPO_SCRIPT_DIR}/../../scripts/_paths.sh"
# <<< capo/scripts/_paths.sh
#CUDA_VISIBLE_DEVICES=0
#bash ./scripts/test_ceo_test.sh V 0 0 100. 5e-3
# bash ./scripts/test_tpt.sh I
# bash ./scripts/test_tpt.sh A
# bash ./scripts/test_tpt.sh R
# bash ./scripts/test_tpt.sh K
 bash ./scripts/test_ceo.sh flower102
 bash ./scripts/test_ceo.sh dtd
 bash ./scripts/test_ceo.sh pets
 bash ./scripts/test_ceo.sh cars
 bash ./scripts/test_ceo.sh ucf101
 bash ./scripts/test_ceo.sh caltech101
 bash ./scripts/test_ceo.sh food101
 bash ./scripts/test_ceo.sh aircraft
 bash ./scripts/test_ceo.sh eurosat
 bash ./scripts/test_ceo.sh sun397
