import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

bandwidths = [5, 10, 50, 100, 1000]

capo_ood = [507.4736938, 495.1118469, 485.2223694, 483.9861847, 482.8736185]
capo_ood_acc=49.92
capo_fgvc = [189.7428394, 177.7564197, 168.1672839, 166.968642, 165.8898642]
capo_fgvc_acc=58.81

ams_ood = [10245.01, 5502.09, 1707.76, 1233.47, 806.61]
ams_ood_acc=49.46
ams_fgvc = [9514.44, 4867.48, 1149.90, 685.21, 266.98]
ams_fgvc_acc=56.75




def plot_latency(capo_data, ams_data, capo_acc, ams_acc, title, filename_base):
    plt.figure(figsize=(3.2, 2.8))

    plt.plot(bandwidths, capo_data, 'o-', label="CAPO")
    plt.plot(bandwidths, ams_data, 's-', label="AMS")

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Bandwidth (Mbps)")
    plt.ylabel("End-to-end Latency (ms)")

    ax = plt.gca()
    ax.set_xticks([5, 10, 100, 1000])
    ax.set_xticklabels(['5', '10', '100', '1000'])
    ax.xaxis.set_minor_locator(ticker.NullLocator())
    ax.yaxis.set_minor_locator(ticker.NullLocator())
    ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
    ax.set_xlim(3, 1500)

    for bw, cv, av in zip(bandwidths, capo_data, ams_data):
        if bw == 5:
            plt.annotate(f'Acc={capo_acc}', (bw, cv), textcoords="offset points",
                         xytext=(6, 4), fontsize=9, color='C0')
            plt.annotate(f'Acc={ams_acc}', (bw, av), textcoords="offset points",
                         xytext=(10, -10), fontsize=9, color='C1')

    plt.legend(loc='upper right')
    plt.grid(True, which='major', linestyle='--', linewidth=0.5, alpha=0.5)
    plt.tight_layout()
    plt.savefig(filename_base + '.png', dpi=300, bbox_inches='tight', pad_inches=0.02)
    plt.savefig(filename_base + '.pdf', bbox_inches='tight', pad_inches=0.02)
    plt.close()


plot_latency(capo_ood, ams_ood, capo_ood_acc, ams_ood_acc,
             "OOD Avg", "figures/latency_ood_avg")

plot_latency(capo_fgvc, ams_fgvc, capo_fgvc_acc, ams_fgvc_acc,
             "FGVC Avg", "figures/latency_fgvc_avg")

print("Saved: latency_ood_avg.png/pdf, latency_fgvc_avg.png/pdf")
