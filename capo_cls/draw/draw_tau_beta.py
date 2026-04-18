import matplotlib
matplotlib.use("Agg")  # 无显示后端
import matplotlib.pyplot as plt

# ===== 数据 =====
tau = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]
perf_tau = [56.9, 57.64, 57.82, 58.26, 57.99, 57.82, 57.73, 56.94, 56.7, 56.49, 56.05, 55.64, 55.21]

beta = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]
perf_beta = [55.04, 55.82, 56.74, 57.32, 57.43, 57.6, 57.76, 58.11, 58.26, 57.99, 57.94, 57.86, 57.86]

# ===== τ 图 =====
plt.figure(figsize=(3.3, 2.5))  # 适合一栏
plt.plot(tau, perf_tau, marker="s", linestyle="-", color="tab:orange")
plt.xlabel(r"$\tau$")
plt.ylabel("Accuracy (%)")
plt.grid(True, linestyle="--", alpha=0.6)
plt.ylim(54.5, 58.5)
plt.tight_layout()
plt.savefig("ablation_tau.png", dpi=300, bbox_inches='tight', pad_inches=0.02)

# ===== β 图 =====
plt.figure(figsize=(3.3, 2.5))  # 适合一栏
plt.plot(beta, perf_beta, marker="o", linestyle="-", color="tab:blue")
plt.xlabel(r"$\beta$")
plt.ylabel("Accuracy (%)")
plt.grid(True, linestyle="--", alpha=0.6)
plt.ylim(54.5, 58.5)
plt.tight_layout()
plt.savefig("ablation_beta.png", dpi=300, bbox_inches='tight', pad_inches=0.02)
