import matplotlib
matplotlib.use("Agg")  # 无显示后端，适合服务器环境
import matplotlib.pyplot as plt
import numpy as np

# ===== 数据 =====
prompt_len = [1, 2, 4, 8, 16]
before = [2.13, 4.13, 8.13, 16.13, 32.13]
after = [1.91, 3.8, 7.16, 14.93, 29.76]

x = np.arange(len(prompt_len))  # X轴位置
bar_width = 0.4

# ===== 作图 =====
fig, ax = plt.subplots(figsize=(5, 3.2))

bars1 = ax.bar(x - bar_width/2, before, bar_width, label="Before", alpha=0.8)
bars2 = ax.bar(x + bar_width/2, after, bar_width, label="After", alpha=0.8)

# 轴标签 & 图例（字号调大）
ax.set_ylabel("Prompt Size (KB)", fontsize=12)
ax.set_xlabel("Prompt Length", fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels(prompt_len, fontsize=11)
ax.legend(fontsize=11)

# 在柱子上标数值
def add_labels(bars):
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f"{height:.2f}",
                    xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 2),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)

add_labels(bars1)
add_labels(bars2)

ymax = max(before + after) * 1.18
ax.set_ylim(0, ymax)

fig.tight_layout()
plt.savefig("prompt_compression_bar.png", dpi=300)
