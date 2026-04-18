#!/bin/bash
# >>> capo/scripts/_paths.sh (auto-inserted)
# Locate scripts/_paths.sh relative to this script's location so the file
# stays portable after a git clone -- no absolute paths.
_CAPO_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
source "${_CAPO_SCRIPT_DIR}/../../scripts/_paths.sh"
# <<< capo/scripts/_paths.sh
# 等待 GPU 全部空闲后自动运行

TARGET_GPUS="0,1,4,7"
SCRIPT_DIR=$(dirname "$(realpath "$0")")
TARGET_SCRIPT="${SCRIPT_DIR}/run_final_ood.sh"
TARGET_ARGS="\"1,0,0,4,7\" 0.1 0.4 1.1"
CHECK_INTERVAL=1800  # 检查间隔（秒）
MEM_THRESHOLD=3500  # 显存阈值（MB），低于此值认为空闲

check_gpus_free() {
    # 检查目标 GPU 是否都空闲（显存使用低于阈值）
    for gpu_id in $(echo $TARGET_GPUS | tr ',' ' '); do
        mem_used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $gpu_id 2>/dev/null)
        if [ -z "$mem_used" ] || [ "$mem_used" -ge "$MEM_THRESHOLD" ]; then
            return 1  # GPU 正在使用
        fi
    done
    return 0  # 所有 GPU 空闲
}

echo "开始监控 GPU $TARGET_GPUS..."
echo "检查间隔: ${CHECK_INTERVAL}秒, 显存阈值: ${MEM_THRESHOLD}MB"
echo "目标脚本: ${TARGET_SCRIPT}"
echo "----------------------------------------"

while true; do
    if check_gpus_free; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] GPU 全部空闲，开始运行脚本..."
        eval bash "$TARGET_SCRIPT" $TARGET_ARGS
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] 脚本已完成，退出监控"
        exit 0
    else
        # 显示当前 GPU 状态
        echo -n "[$(date '+%Y-%m-%d %H:%M:%S')] GPU 显存使用(MB): "
        nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits -i $TARGET_GPUS | tr '\n' ' '
        echo "- 等待中..."
    fi
    sleep $CHECK_INTERVAL
done
