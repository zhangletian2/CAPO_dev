#!/bin/bash
# >>> capo/scripts/_paths.sh (auto-inserted)
# Locate scripts/_paths.sh relative to this script's location so the file
# stays portable after a git clone -- no absolute paths.
_CAPO_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
source "${_CAPO_SCRIPT_DIR}/../../scripts/_paths.sh"
# <<< capo/scripts/_paths.sh
# ─────────────────────────────────────────────────────────────
#  CAPO E2E Benchmark — 全数据集 × 4 种 uplink_percent, 4 GPU 并行
#
#  15 datasets × 4 uplink% = 60 jobs, round-robin across GPUs.
#  Each job: 500 samples ≈ 8 min → total ≈ 2 hours on 4 GPUs.
#
#  Usage:
#    bash scripts/run_e2e_bench.sh [gpu_list]
#
#  Examples:
#    bash scripts/run_e2e_bench.sh "0,1,2,3"    # GPUs 0-3
#    bash scripts/run_e2e_bench.sh "4,5,6,7"    # GPUs 4-7
# ─────────────────────────────────────────────────────────────

set -e

GPU_LIST=${1:-"4,5,6,7"}
IFS=',' read -ra GPUS <<< "${GPU_LIST}"
NUM_GPUS=${#GPUS[@]}

# ── config ──
data_root='./dataset'
max_samples=500
arch=RN50
cloud_arch=ViT-B/16
ctx_init=a_photo_of_a
selection_p=0.1
tau=0.4
beta=0.9
rec_dir=new
out_dir=results/e2e_bench
log_dir=logs/e2e_bench

mkdir -p "${out_dir}" "${log_dir}"

DATASETS=(I A R K V flower102 dtd eurosat cars pets food101 ucf101 sun397 aircraft caltech101)
UPLINKS=(25 50 75 100)

total=$(( ${#DATASETS[@]} * ${#UPLINKS[@]} ))
echo "═══════════════════════════════════════════════════════════"
echo "  CAPO E2E Benchmark"
echo "  ${#DATASETS[@]} datasets × ${#UPLINKS[@]} uplink% = ${total} jobs"
echo "  GPUs: ${GPU_LIST}  |  max_samples: ${max_samples}  |  rec_dir: ${rec_dir}"
echo "═══════════════════════════════════════════════════════════"

job_idx=0
for up in "${UPLINKS[@]}"; do
    for ds in "${DATASETS[@]}"; do
        gpu=${GPUS[$((job_idx % NUM_GPUS))]}
        csv="${out_dir}/bench_${ds}_up${up}.csv"
        log="${log_dir}/${ds}_up${up}.log"

        echo "[$(printf '%2d' $((job_idx+1)))/${total}]  GPU ${gpu}  ${ds}  uplink=${up}%"

        python ./ceo_e2e_bench.py ${data_root} \
            --test_sets ${ds} \
            -a ${arch} -c ${cloud_arch} -b 64 --gpu ${gpu} \
            --ctx_init ${ctx_init} \
            --selection_p ${selection_p} --tau ${tau} --beta ${beta} \
            --uplink_percent ${up} \
            --rec_dir ${rec_dir} \
            --output_csv "${csv}" \
            --max_samples ${max_samples} \
            --warmup 10 \
            > "${log}" 2>&1 &

        job_idx=$((job_idx + 1))
        if (( job_idx % NUM_GPUS == 0 )); then
            wait
        fi
    done
done
wait

# ── collect summary from logs ──
echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  All ${total} jobs done.  Summary:"
echo "═══════════════════════════════════════════════════════════"
printf "%-12s" "Dataset"
for up in "${UPLINKS[@]}"; do
    printf "  up%-4s(sync/async)" "${up}%"
done
echo ""
echo "────────────────────────────────────────────────────────────────────────────────────"
for ds in "${DATASETS[@]}"; do
    printf "%-12s" "${ds}"
    for up in "${UPLINKS[@]}"; do
        log="${log_dir}/${ds}_up${up}.log"
        if [ -f "${log}" ]; then
            sync_acc=$(grep "Sync  (post-TTA):" "${log}" 2>/dev/null | awk '{print $NF}' | tail -1)
            async_acc=$(grep "Async (pre-TTA):" "${log}" 2>/dev/null | awk '{print $NF}' | tail -1)
            printf "  %s / %s     " "${sync_acc:-N/A}" "${async_acc:-N/A}"
        else
            printf "  %-21s" "N/A"
        fi
    done
    echo ""
done

echo ""
echo "CSVs  → ${out_dir}/"
echo "Logs  → ${log_dir}/"
echo "Detail: tail ${log_dir}/<ds>_up<pct>.log"
