#!/bin/bash
# >>> capo/scripts/_paths.sh (auto-inserted)
# Locate scripts/_paths.sh relative to this script's location so the file
# stays portable after a git clone -- no absolute paths.
_CAPO_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
source "${_CAPO_SCRIPT_DIR}/../../scripts/_paths.sh"
# <<< capo/scripts/_paths.sh
bash ./scripts/test_ams.sh flower102
bash ./scripts/test_ams.sh dtd
bash ./scripts/test_ams.sh pets
bash ./scripts/test_ams.sh cars
bash ./scripts/test_ams.sh ucf101
bash ./scripts/test_ams.sh caltech101
bash ./scripts/test_ams.sh food101
bash ./scripts/test_ams.sh aircraft
bash ./scripts/test_ams.sh eurosat
bash ./scripts/test_ams.sh sun397
#bash ./scripts/test_ams.sh I
#bash ./scripts/test_ams.sh A
#bash ./scripts/test_ams.sh V
#bash ./scripts/test_ams.sh R
#bash ./scripts/test_ams.sh K