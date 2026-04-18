#!/usr/bin/env bash
# ----------------------------------------------------------------------
# capo - one-shot enc2dec preprocessing wrapper.
#
# Runs the cls preprocessing (14 datasets) and the ret preprocessing
# (COCO2014 karpathy_test) at QP=1 to produce the `new/` and
# `coco2014_cloud/val2014/` folders that CEO reads at runtime.
#
# QP=7 variants (new_qp7/, *_cloud_qp7/) are intentionally NOT produced.
# You only need to run this ONCE per clone after setup.sh.
#
# Prerequisites:
#   * conda activate joint3
#   * datasets/{cls,ret} symlinks populated by scripts/setup.sh
#   * enc2dec/models/model0_qp1.pk present (see ASSETS.md)
#   * enc2dec/module_arithmeticcoding compiled (setup.sh does this)
#
# Usage:
#   bash scripts/preprocess_enc2dec.sh [cls|ret|all] [device|N_SHARDS] [ret_gpu_list]
#
# Examples:
#   bash scripts/preprocess_enc2dec.sh all                     # cls on cuda:0, ret on GPU 0-3
#   bash scripts/preprocess_enc2dec.sh cls cuda:2              # cls only, on GPU 2
#   bash scripts/preprocess_enc2dec.sh ret 4 "2,3,4,5"         # ret only, 4 shards on GPU 2-5
# ----------------------------------------------------------------------
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_paths.sh"

WHAT=${1:-all}

case "${WHAT}" in
    cls)
        DEVICE=${2:-cuda:0}
        bash "${ENC2DEC_ROOT}/scripts/preprocess_cls.sh" "${DEVICE}"
        ;;
    ret)
        N_SHARDS=${2:-4}
        GPU_LIST=${3:-"0,1,2,3"}
        bash "${ENC2DEC_ROOT}/scripts/preprocess_ret.sh" "${N_SHARDS}" "${GPU_LIST}"
        ;;
    all)
        DEVICE=${2:-cuda:0}
        N_SHARDS=${3:-4}
        GPU_LIST=${4:-"0,1,2,3"}
        echo ">>> (1/2) cls preprocessing"
        bash "${ENC2DEC_ROOT}/scripts/preprocess_cls.sh" "${DEVICE}"
        echo ">>> (2/2) ret preprocessing"
        bash "${ENC2DEC_ROOT}/scripts/preprocess_ret.sh" "${N_SHARDS}" "${GPU_LIST}"
        ;;
    *)
        echo "Unknown target: ${WHAT}" >&2
        echo "Usage: $0 [cls|ret|all] ..." >&2
        exit 1
        ;;
esac
