import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import os

OUT_DIR = "figures"
os.makedirs(OUT_DIR, exist_ok=True)

# ================== 数据 ==================
x_ceo_ood = np.array([3.95579, 7.91158, 11.86737, 15.82316])
y_ceo_ood = np.array([47.53, 49.92, 51.50, 52.08])
x_ams_ood = np.array([1517.73, 3035.47, 4553.20, 6070.93])
y_ams_ood = np.array([45.50, 46.82, 48.14, 49.46])

x_ceo_fgvc = np.array([3.835655, 7.67131, 11.506965, 15.34262])
y_ceo_fgvc = np.array([57.56, 58.81, 59.46, 62.10])
x_ams_fgvc = np.array([1487.03, 2974.06, 4461.09, 5948.12])
y_ams_fgvc = np.array([56.05, 56.29, 56.52, 56.75])

x_ceo_t2i = np.array([1.77007, 3.54014, 5.31021, 7.0803])
y_ceo_t2i = np.array([29.37, 30.35, 31.29, 33.18])
x_ams_t2i = np.array([1353.26, 2706.53, 4059.79, 5413.05])
y_ams_t2i = np.array([29.28, 30.50, 31.73, 32.95])

x_ceo_i2t = np.array([4.251505, 8.50301, 12.754515, 17.00602])
y_ceo_i2t = np.array([48.56, 50.96, 53.06, 53.74])
x_ams_i2t = np.array([1577.73, 3155.46, 4733.19, 6310.92])
y_ams_i2t = np.array([50.18, 50.92, 51.66, 52.40])

labels = ["k=25%", "k=50%", "k=75%", "k=100%"]

# ================== 统一布局参数 ==================
FIGSIZE = (3.2, 2.8)
MARGINS = dict(left=0.19, right=0.96, bottom=0.17, top=0.98)

# ================== 通用绘制函数 ==================
def plot_curve(x_ceo, y_ceo, x_ams, y_ams, ylabel, stem, ylim=None, legend_loc='upper right'):
    fig, ax = plt.subplots(figsize=FIGSIZE)
    fig.subplots_adjust(**MARGINS)

    ax.plot(x_ceo, y_ceo, 'o-', label="CAPO")
    ax.plot(x_ams, y_ams, 's-', label="AMS")

    for i in (0, len(labels) - 1):
        ax.annotate(labels[i], (x_ceo[i], y_ceo[i]), textcoords="offset points",
                    xytext=(5, 0), ha='left', va='center', fontsize=9)
    for i in (0, len(labels) - 1):
        ax.annotate(labels[i], (x_ams[i], y_ams[i]), textcoords="offset points",
                    xytext=(-5, 0), ha='right', va='center', fontsize=9)

    ax.set_xscale("log")
    ax.set_xlim(1, 10000)
    ax.set_xticks([1, 10, 100, 1000, 10000])
    ax.set_xticklabels(['$10^0$', '$10^1$', '$10^2$', '$10^3$', '$10^4$'])
    ax.xaxis.set_minor_locator(ticker.NullLocator())
    ax.set_xlabel("Transmission Cost (KB)")
    ax.set_ylabel(ylabel)
    if ylim:
        ax.set_ylim(ylim)
    ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=4))
    ax.legend(loc=legend_loc)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.5)

    for fmt in ("png", "pdf"):
        fig.savefig(os.path.join(OUT_DIR, f"{stem}.{fmt}"), dpi=300)
    plt.close(fig)


# ================== 分别绘制 ==================
plot_curve(x_ceo_ood, y_ceo_ood, x_ams_ood, y_ams_ood,
           ylabel="Acc@1", stem="eff_curve_ood_acc",
           ylim=(44.5, 53), legend_loc='upper right')

plot_curve(x_ceo_fgvc, y_ceo_fgvc, x_ams_fgvc, y_ams_fgvc,
           ylabel="Acc@1", stem="eff_curve_fgvc_acc",
           ylim=(55.5, 63), legend_loc='upper right')

plot_curve(x_ceo_t2i, y_ceo_t2i, x_ams_t2i, y_ams_t2i,
           ylabel="R@1", stem="eff_curve_t2i_r1",
           ylim=(27.3, 34.2), legend_loc='lower right')

plot_curve(x_ceo_i2t, y_ceo_i2t, x_ams_i2t, y_ams_i2t,
           ylabel="R@1", stem="eff_curve_i2t_r1",
           ylim=(47.7, 54.7), legend_loc='lower right')
