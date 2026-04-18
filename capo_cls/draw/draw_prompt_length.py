# -*- coding: utf-8 -*-
import matplotlib
matplotlib.use("Agg")  # 无显示后端
import matplotlib.pyplot as plt
import numpy as np


def plot_prompt_length_tradeoff(m, acc, kb, save_path="prompt_len_tradeoff",
                                fig_size=(3.35, 2.4)):
    """
    左图：Accuracy（左轴，折线） + Transmission（右轴，条形）
    - 横轴等间距
    - 左轴纵坐标显示为整数刻度，3~4 个，顶部留白
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.ticker import MaxNLocator

    labels = [str(v) for v in m]
    x = np.arange(len(m))

    fig, ax_acc = plt.subplots(figsize=fig_size)
    ax_kb = ax_acc.twinx()

    bar_width = 0.55
    ax_kb.bar(
        x, kb, width=bar_width,
        alpha=0.25, edgecolor="tab:orange", color="tab:orange",
        zorder=1, label="Transmission (KB)"
    )
    ax_kb.set_ylabel("Transmission volume (KB)")
    ax_kb.set_ylim(0, np.max(kb) * 1.15)
    ax_kb.margins(x=0.02)

    ax_acc.plot(
        x, acc, marker="o", linewidth=1.8,
        color="tab:blue", zorder=3, label="Accuracy (%)"
    )
    ax_acc.set_xlabel(r"Prompt length $m$")
    ax_acc.set_ylabel("Accuracy (%)")
    ax_acc.set_xticks(x)
    ax_acc.set_xticklabels(labels)

    ymin = int(np.floor(np.min(acc))) - 1
    ymax = int(np.ceil(np.max(acc))) + 1.5
    ax_acc.set_ylim(ymin, ymax)
    ax_acc.yaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))

    ax_acc.grid(True, axis="y", linestyle="--", alpha=0.3, zorder=0)

    h1, l1 = ax_acc.get_legend_handles_labels()
    h2, l2 = ax_kb.get_legend_handles_labels()
    ax_acc.legend(
        h1 + h2, l1 + l2,
        loc="lower center", bbox_to_anchor=(0.5, 1.02),
        ncol=2, fontsize=9, frameon=False
    )

    plt.subplots_adjust(top=0.87)
    save_path = save_path.removesuffix(".png").removesuffix(".pdf")
    fig.savefig(f"{save_path}.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{save_path}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_gain_per_kb(m, acc, kb, save_path="prompt_len_gain_per_kb",
                     fig_size=(3.35, 2.4)):
    """
    右图：ΔAcc / ΔKB（相对 m=1）
    - 自动给 y 轴上限增加 padding，避免数值标注越界
    - 若柱子太高，标注会被“夹”在上限内，始终可见
    """
    base_acc, base_kb = acc[0], kb[0]
    m_rel = m[1:]
    gain_per_kb = (acc[1:] - base_acc) / (kb[1:] - base_kb)

    fig, ax = plt.subplots(figsize=fig_size)
    bar_width = 0.55
    bars = ax.bar([str(v) for v in m_rel], gain_per_kb,
                  color="tab:green", width=bar_width)

    ax.set_xlabel(r"Prompt length $m$")
    # ax.set_ylabel(r"$\Delta$Acc / $\Delta$KB")
    ax.set_ylabel(r"$\Delta$R@1 / $\Delta$KB")

    ax.grid(True, axis="y", linestyle="--", alpha=0.3)

    # —— 顶部留白，避免标注越界；再把标注位置“夹”在坐标轴范围内
    ymax = float(np.max(gain_per_kb)) * 1.20
    ymin = min(0.0, float(np.min(gain_per_kb)) * 1.10)
    ax.set_ylim(ymin, ymax)

    for rect, val in zip(bars, gain_per_kb):
        # 计算理想标注位置；若超过轴上限，则“夹紧”到 (ymax - margin)
        ideal_y = float(val) + (ymax - ymin) * 0.03
        clipped_y = min(ideal_y, ymax - (ymax - ymin) * 0.03)
        if clipped_y < 0:
            clipped_y = 0.01
        ax.annotate(
            f"{val:.2f}",
            xy=(rect.get_x() + rect.get_width() / 2.0, clipped_y),
            xytext=(0, 0),
            textcoords="offset points",
            ha="center", va="bottom",
            fontsize=9, color="black"
        )

    fig.tight_layout(pad=0.2)
    save_path = save_path.removesuffix(".png").removesuffix(".pdf")
    fig.savefig(f"{save_path}.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{save_path}.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    # ===== 示例数据 =====
    m = np.array([1, 2, 4, 8, 16])
    # acc = np.array([53.95, 54.93, 58.26, 57.93, 57.13])
    kb = np.array([7.639492188, 9.529492188, 12.88949219, 20.65949219, 35.48949219])
    acc = np.array([30.48, 31.06, 33.18, 31.21, 29.16])
    # acc = np.array([50.76, 52.66, 53.74, 53.96, 52.36])

    import os
    fig_dir = os.path.join(os.path.dirname(__file__), "..", "figures")
    os.makedirs(fig_dir, exist_ok=True)
    # plot_prompt_length_tradeoff(m, acc, kb, save_path=os.path.join(fig_dir, "prompt_len_tradeoff"))
    # plot_gain_per_kb(m, acc, kb, save_path=os.path.join(fig_dir, "prompt_len_gain_per_kb"))
    plot_gain_per_kb(m, acc, kb, save_path=os.path.join(fig_dir, "prompt_len_gain_per_kb_ret"))

