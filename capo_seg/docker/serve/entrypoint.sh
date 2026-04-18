#!/bin/bash
# >>> capo/scripts/_paths.sh (auto-inserted)
# Locate scripts/_paths.sh relative to this script's location so the file
# stays portable after a git clone -- no absolute paths.
_CAPO_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
source "${_CAPO_SCRIPT_DIR}/../../../scripts/_paths.sh"
# <<< capo/scripts/_paths.sh
set -e

if [[ "$1" = "serve" ]]; then
    shift 1
    torchserve --start --ts-config /home/model-server/config.properties
else
    eval "$@"
fi

# prevent docker exit
tail -f /dev/null
