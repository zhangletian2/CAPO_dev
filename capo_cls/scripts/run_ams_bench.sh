#!/bin/bash
# >>> capo/scripts/_paths.sh (auto-inserted)
# Locate scripts/_paths.sh relative to this script's location so the file
# stays portable after a git clone -- no absolute paths.
_CAPO_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
source "${_CAPO_SCRIPT_DIR}/../../scripts/_paths.sh"
# <<< capo/scripts/_paths.sh
# ─────────────────────────────────────────────────────────────
#  AMS E2E Benchmark — 全数据集, 4 GPU 并行
#
#  AMS 没有 uplink_percent (所有样本都上传),
#  所以 15 datasets = 15 jobs, round-robin across GPUs.
#
#  Usage:
#    bash scripts/run_ams_bench.sh [gpu_list]
#
#  Examples:
#    bash scripts/run_ams_bench.sh "0,1,2,3"
#    bash scripts/run_ams_bench.sh "4,5,6,7"
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
lr=1e-6
rec_dir=new
out_dir=results/ams_bench
log_dir=logs/ams_bench

mkdir -p "${out_dir}" "${log_dir}"

DATASETS=(I A R K V flower102 dtd eurosat cars pets food101 ucf101 sun397 aircraft caltech101)

total=${#DATASETS[@]}
echo "═══════════════════════════════════════════════════════════"
echo "  AMS E2E Benchmark"
echo "  ${total} datasets  |  GPUs: ${GPU_LIST}"
echo "  max_samples: ${max_samples}  |  lr: ${lr}  |  rec_dir: ${rec_dir}"
echo "═══════════════════════════════════════════════════════════"

job_idx=0
for ds in "${DATASETS[@]}"; do
    gpu=${GPUS[$((job_idx % NUM_GPUS))]}
    csv_file="${out_dir}/bench_${ds}.csv"
    log_file="${log_dir}/${ds}.log"

    echo "[$(printf '%2d' $((job_idx+1)))/${total}]  GPU ${gpu}  ${ds}"

    python ./ams_e2e_bench.py ${data_root} \
        --test_sets ${ds} \
        -a ${arch} -c ${cloud_arch} -b 64 --gpu ${gpu} \
        --ctx_init ${ctx_init} \
        --lr ${lr} \
        --rec_dir ${rec_dir} \
        --output_csv "${csv_file}" \
        --max_samples ${max_samples} \
        --warmup 10 \
        > "${log_file}" 2>&1 &

    job_idx=$((job_idx + 1))
    if (( job_idx % NUM_GPUS == 0 )); then
        wait
    fi
done
wait

# ── collect summary ──
echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  All ${total} jobs done.  Summary:"
echo "═══════════════════════════════════════════════════════════"
printf "%-14s  %8s  %10s  %10s  %10s  %10s\n" \
    "Dataset" "Acc(%)" "Cloud(ms)" "Distill(ms)" "Final(ms)" "Total(ms)"
echo "──────────────────────────────────────────────────────────────────────────"
for ds in "${DATASETS[@]}"; do
    log_file="${log_dir}/${ds}.log"
    if [ -f "${log_file}" ]; then
        acc=$(grep "Post-TTA:" "${log_file}" 2>/dev/null | awk '{print $2}' | tail -1)
        cloud=$(grep "Cloud inference:" "${log_file}" 2>/dev/null | awk '{print $3}' | tail -1)
        distill=$(grep "KL distillation:" "${log_file}" 2>/dev/null | awk '{print $3}' | tail -1)
        final=$(grep "Edge final infer:" "${log_file}" 2>/dev/null | awk '{print $4}' | tail -1)
        total_c=$(grep "Total compute:" "${log_file}" 2>/dev/null | awk '{print $NF}' | tail -1)
        printf "%-14s  %8s  %10s  %11s  %10s  %10s\n" \
            "${ds}" "${acc:-N/A}" "${cloud:-N/A}" "${distill:-N/A}" \
            "${final:-N/A}" "${total_c:-N/A}"
    else
        printf "%-14s  %8s\n" "${ds}" "N/A"
    fi
done

echo ""
echo "CSVs  → ${out_dir}/"
echo "Logs  → ${log_dir}/"
echo "Detail: cat ${log_dir}/<ds>.log"
