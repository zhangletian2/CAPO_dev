#!/usr/bin/env bash
# ----------------------------------------------------------------------
# capo — one-shot setup after `git clone`.
#
# Reads .env (copy from .env.example) and:
#   1. Creates datasets/{cls,ret,seg}/ symlinks to user-provided roots.
#   2. Wires up per-task artifact dirs (clip_cache, pretrain, pretrained,
#      saved_params_*, etc.) as symlinks if the matching *_SRC is set,
#      otherwise creates empty real dirs.
#   3. Compiles enc2dec/module_arithmeticcoding from the .cpp source.
#   4. Prints a sanity-check summary.
#
# Usage:
#   cp .env.example .env && vim .env
#   conda activate joint3
#   bash scripts/setup.sh
# ----------------------------------------------------------------------
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

if [ ! -f .env ]; then
    echo "ERROR: .env not found. Copy .env.example to .env and edit it first." >&2
    exit 1
fi

# Export all vars defined in .env.
set -a; . ./.env; set +a

# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
link_if() {
    # link_if <target_path> <link_path>
    # If the target exists, (re)create a symlink at link_path pointing to it.
    # If not, create an empty real dir at link_path (but don't overwrite
    # anything the user already has there).
    # Always writes an absolute-path symlink so the link resolves correctly
    # regardless of the CWD from which it is followed.
    local tgt="$1" dst="$2"
    if [ -e "${dst}" ] && [ ! -L "${dst}" ]; then
        echo "  keep  ${dst}  (pre-existing non-symlink, left alone)"
        return
    fi
    [ -L "${dst}" ] && rm "${dst}"
    if [ -n "${tgt}" ] && [ -e "${tgt}" ]; then
        # Resolve to absolute path for a robust link
        local abs_tgt
        abs_tgt="$(cd "${tgt}" 2>/dev/null && pwd || readlink -f "${tgt}")"
        ln -s "${abs_tgt}" "${dst}"
        echo "  link  ${dst}  ->  ${abs_tgt}"
    else
        mkdir -p "${dst}"
        echo "  mkdir ${dst}  (placeholder; set the *_SRC var in .env to link to real data)"
    fi
}

header() { echo; echo "=== $* ==="; }

# ----------------------------------------------------------------------
# 1) Datasets
# ----------------------------------------------------------------------
header "Datasets"
mkdir -p datasets
link_if "${CLS_DATA_SRC:-}"  datasets/cls
link_if "${RET_DATA_SRC:-}"  datasets/ret
link_if "${SEG_DATA_SRC:-}"  datasets/seg

# In-task dataset aliases (so relative paths like ./dataset keep working)
link_if datasets/cls capo_cls/dataset
link_if datasets/ret capo_ret/dataset
link_if datasets/seg capo_seg/data

# ----------------------------------------------------------------------
# 2) Pretrained weights
# ----------------------------------------------------------------------
header "Pretrained weights"
link_if "${CLIP_CACHE_SRC:-}"      capo_cls/clip_cache
link_if "${SEG_PRETRAIN_SRC:-}"    capo_seg/pretrain
link_if "${ENC2DEC_MODELS_SRC:-}"  enc2dec/models

# CLIP weights are shared between cls and ret.  The LAVIS loader
# (third_party/lavis/models/clip_models/pretrained.py) reads from
# RET_CLIP_CACHE which defaults to CLIP_CACHE (set above).
# We still need capo_ret/pretrained/clip/ as a directory structure
# so that legacy --cfg-path YAML references resolve.
# Strategy:
#   - If RET_PRETRAINED_SRC is explicitly set, use it for the whole
#     capo_ret/pretrained tree (user has their own separate copy).
#   - Otherwise, create capo_ret/pretrained/ as a real dir and
#     put a symlink `clip/` inside it pointing at capo_cls/clip_cache
#     (the single shared CLIP weights).
if [ -n "${RET_PRETRAINED_SRC:-}" ]; then
    link_if "${RET_PRETRAINED_SRC}" capo_ret/pretrained
else
    link_if "" capo_ret/pretrained             # mkdir -p
    # Resolve the CLIP cache absolute path for the inner symlink
    _clip_abs="$(cd capo_cls/clip_cache 2>/dev/null && pwd || echo "")"
    if [ -n "${_clip_abs}" ]; then
        [ -L capo_ret/pretrained/clip ] && rm capo_ret/pretrained/clip
        ln -s "${_clip_abs}" capo_ret/pretrained/clip
        echo "  link  capo_ret/pretrained/clip  ->  ${_clip_abs}  (shared with capo_cls/clip_cache)"
    else
        echo "  skip  capo_ret/pretrained/clip  (CLIP_CACHE_SRC not set yet; re-run after setting it)"
    fi
fi

# ----------------------------------------------------------------------
# 3) Artifact dirs (logs / results / checkpoints / saved_params_* / ...)
#    If the matching *_OUTPUT_SRC is set in .env we redirect to it,
#    otherwise create real dirs in-tree.
# ----------------------------------------------------------------------
header "Artifact dirs"
mkdir -p capo_joint/work_dirs capo_seg/work_dirs

for name in logs results checkpoints saved_params_ams saved_prompts_nctx; do
    if [ -n "${CLS_OUTPUT_SRC:-}" ]; then
        link_if "${CLS_OUTPUT_SRC}/${name}" "capo_cls/${name}"
    else
        link_if "" "capo_cls/${name}"
    fi
done

for name in logs output saved_params_ams_ret saved_prompts_ret; do
    # pretrained is handled in the "Pretrained weights" section above
    if [ -n "${RET_OUTPUT_SRC:-}" ]; then
        link_if "${RET_OUTPUT_SRC}/${name}" "capo_ret/${name}"
    else
        link_if "" "capo_ret/${name}"
    fi
done

for name in logs; do
    if [ -n "${SEG_OUTPUT_SRC:-}" ]; then
        link_if "${SEG_OUTPUT_SRC}/${name}" "capo_seg/${name}"
    else
        link_if "" "capo_seg/${name}"
    fi
done

for name in logs intermediate intermediate_qp7; do
    if [ -n "${ENC2DEC_OUTPUT_SRC:-}" ]; then
        link_if "${ENC2DEC_OUTPUT_SRC}/${name}" "enc2dec/${name}"
    else
        link_if "" "enc2dec/${name}"
    fi
done

# ----------------------------------------------------------------------
# 4) Compile the arithmetic-coding native helper (from .cpp)
# ----------------------------------------------------------------------
header "Compile enc2dec/module_arithmeticcoding"
if [ -x enc2dec/module_arithmeticcoding ]; then
    echo "  already compiled, skipping"
else
    (
        cd enc2dec
        if command -v g++ >/dev/null 2>&1; then
            g++ module_arithmeticcoding.cpp -o module_arithmeticcoding \
                && echo "  built enc2dec/module_arithmeticcoding"
        else
            echo "  WARNING: g++ not in PATH; arithmetic coder NOT built."
            echo "           enc2dec can still run, but encode/decode steps will fail."
        fi
    )
fi

# ----------------------------------------------------------------------
# 5) Summary
# ----------------------------------------------------------------------
header "Sanity check"
python - <<'PY' || true
import os, sys
root = os.environ.get('REPO_ROOT', os.getcwd())
sys.path.insert(0, os.path.join(root, 'third_party'))
ok = True
for mod in ('clip', 'lavis', 'mmseg'):
    try:
        m = __import__(mod)
        print(f"  import {mod:6s}  OK   ({m.__file__})")
    except ImportError as e:
        print(f"  import {mod:6s}  FAIL ({e})")
        ok = False
sys.exit(0 if ok else 1)
PY

header "Symlink status"
for p in datasets/cls datasets/ret datasets/seg \
         capo_cls/clip_cache capo_cls/logs capo_cls/checkpoints \
         capo_ret/pretrained capo_ret/logs \
         capo_seg/pretrain capo_seg/data \
         enc2dec/models enc2dec/intermediate_qp7; do
    if [ -e "${p}" ]; then
        tgt=$(readlink -f "${p}" 2>/dev/null || echo "<real dir>")
        printf "  OK  %-40s  ->  %s\n" "${p}" "${tgt}"
    else
        printf "  --  %-40s  (missing, edit .env and re-run)\n" "${p}"
    fi
done

echo
echo "Setup done. Next steps:"
echo "  conda activate joint3"
echo "  source scripts/_paths.sh"
echo "  cd capo_cls && python ceo.py ...    # or any other task"
