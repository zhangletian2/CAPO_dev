# Unified path definitions for the capo repo.
# All task shell scripts source this file (auto-injected at the top).

# -- REPO_ROOT is the directory containing this file's parent ----------------
# Resolve it relative to this file so the repo works at any clone location,
# unless the user has already exported REPO_ROOT explicitly.
if [ -z "${REPO_ROOT:-}" ]; then
    _this_dir="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
    export REPO_ROOT="$(cd "${_this_dir}/.." && pwd)"
    unset _this_dir
fi

# -- Load per-clone overrides from .env (user-specific, git-ignored) ---------
if [ -f "${REPO_ROOT}/.env" ]; then
    # shellcheck disable=SC1091
    set -a; . "${REPO_ROOT}/.env"; set +a
fi

# -- Per-task source roots ---------------------------------------------------
export CLS_ROOT="${REPO_ROOT}/capo_cls"
export RET_ROOT="${REPO_ROOT}/capo_ret"
export SEG_ROOT="${REPO_ROOT}/capo_seg"
export ENC2DEC_ROOT="${REPO_ROOT}/enc2dec"
export JOINT_ROOT="${REPO_ROOT}/capo_joint"
export LAVIS_DIR="${REPO_ROOT}/third_party/lavis"

# -- Dataset roots (symlinks created by setup.sh) ----------------------------
export DATA_ROOT="${REPO_ROOT}/datasets"
export DATA_CLS="${DATA_ROOT}/cls"
export DATA_RET="${DATA_ROOT}/ret"
export DATA_SEG="${DATA_ROOT}/seg"

# -- CLIP weights cache (shared by cls + ret) --------------------------------
# A single copy of the OpenAI CLIP weights (RN50.pt, RN101.pt, ViT-B-16.pt,
# ViT-L-14.pt) is stored here. Both cls (`clip.load(download_root=...)`) and
# ret (LAVIS `pretrained.py`) read from the same directory.
# setup.sh creates a symlink at `capo_ret/pretrained/clip` -> CLIP_CACHE so
# that LAVIS's relative-path loader also resolves correctly.
export CLIP_CACHE="${CLIP_CACHE:-${CLS_ROOT}/clip_cache}"
export TORCH_HOME="${TORCH_HOME:-${CLS_ROOT}}"
export RET_CLIP_CACHE="${RET_CLIP_CACHE:-${CLIP_CACHE}}"

# -- GPUs --------------------------------------------------------------------
# Override in your .env if needed.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2,3,4,5,6,7}"

# -- PYTHONPATH so vendored third_party and task dirs are importable ---------
# `third_party` first so `import clip / lavis / mmseg` resolves to the
# vendored copies; task roots after so flat imports like
# `from data.x import ...` work when running entry scripts with relative CWD.
export PYTHONPATH="${REPO_ROOT}/third_party:${CLS_ROOT}:${RET_ROOT}:${SEG_ROOT}:${ENC2DEC_ROOT}:${JOINT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
