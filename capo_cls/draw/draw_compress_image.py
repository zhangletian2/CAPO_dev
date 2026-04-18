import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ===== 数据 =====
datasets = ["ImageNet-V2"]
before = [149.16]
after = [19.25]

x = np.arange(len(datasets))  # X轴位置
bar_width = 0.22  # 柱子宽度更细

# ===== 作图 =====
fig, ax = plt.subplots(figsize=(3.0, 2.4))  # 单栏子图常用尺寸

bars1 = ax.bar(x - bar_width/2, before, bar_width, label="Before", color="#6F4E7C", alpha=0.9)
bars2 = ax.bar(x + bar_width/2, after,  bar_width, label="After",  color="#54A24B", alpha=0.9)

# 标签 & 坐标轴
ax.set_ylabel("Average Size (KB)")
ax.set_xticks(x)
ax.set_xticklabels(datasets)
ax.legend(frameon=True, fontsize=9, loc="upper right")

# 顶部数值
def add_labels(bars):
    for bar in bars:
        h = bar.get_height()
        ax.annotate(f"{h:.2f}",
                    xy=(bar.get_x() + bar.get_width()/2, h),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center", va="bottom", fontsize=9)
add_labels(bars1)
add_labels(bars2)

# —— 留白：两边各留 0.4 ——
ax.set_xlim(-0.5, 0.5)  # 保证只有一个类时不会贴边
ymax = max(before + after) * 1.18
ax.set_ylim(0, ymax)

plt.tight_layout(pad=0.4)
plt.savefig("image_compression_bar.png", dpi=300, bbox_inches="tight")
