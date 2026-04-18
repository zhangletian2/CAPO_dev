#!/usr/bin/env python3
"""
gen_ams_tables.py — 汇总 AMS 基准测试结果, OOM 数据集用比例估算

OOM 数据集: I, K, V (1000 类, text_encoder 过大)
估算方法: 以 A, R (200 类 OOD) 的 AMS/CAPO 比例为基准,
         对 I, K, V 的每个 timing 分量按 CAPO 值等比推算.

输出: CSV + 文本汇总表 (AMS 各项延迟 + 与 CAPO 对比)
"""

import csv
import os
import re
import numpy as np

# ─── constants ───

BANDWIDTHS = [1000, 100, 50, 10, 5, 1]

OOD = ["I", "A", "R", "K", "V"]
FEWSHOT = ["flower102", "dtd", "eurosat", "cars", "pets",
           "food101", "ucf101", "sun397", "aircraft", "caltech101"]
ALL_DS = OOD + FEWSHOT

DS_DISPLAY = {
    "I": "ImageNet", "A": "ImageNet-A", "R": "ImageNet-R",
    "K": "ImageNet-K", "V": "ImageNet-V2",
}

# AMS downloads entire text_encoder: 149974 KB = ~146.46 MB
AMS_DOWNLOAD_KB = 149974.00

# QP1 bitstream (same for both methods)
BITSTREAM_KB = {
    "I": 7.88, "A": 6.40, "R": 6.63, "K": 13.13, "V": 7.13,
    "flower102": 5.68, "dtd": 8.85, "eurosat": 0.13, "cars": 8.03,
    "pets": 4.76, "food101": 4.24, "ucf101": 1.80, "sun397": 25.23,
    "aircraft": 14.60, "caltech101": 3.02,
}

# CAPO prompt download
CAPO_DOWNLOAD_KB = 3.99

# Image enc/dec times (from image_enc_dec_times.csv, same for both)
ENC_DEC_MS = {
    "I": (118, 290), "A": (103, 246), "R": (144, 364),
    "K": (331, 891), "V": (123, 303),
    "flower102": (162, 415), "dtd": (122, 301), "eurosat": (25, 27),
    "cars": (252, 667), "pets": (103, 248), "food101": (116, 282),
    "ucf101": (56, 112), "sun397": (521, 1428), "aircraft": (366, 992),
    "caltech101": (61, 127),
}

OOM_DATASETS = {"I", "K", "V"}
REFERENCE_DATASETS = ["A", "R"]  # 200-class OOD, used to compute AMS/CAPO ratio


# ─── parse AMS bench logs ───

def parse_ams_log(path):
    d = {}
    with open(path) as f:
        text = f.read()
    if "OutOfMemoryError" in text:
        d["oom"] = True
        return d
    d["oom"] = False

    def _f(pat):
        m = re.search(pat, text)
        return float(m.group(1)) if m else 0.0

    d["acc"] = _f(r"Post-TTA:\s+([\d.]+)%")
    d["t_cloud"] = _f(r"Cloud inference:\s+([\d.]+)")
    d["t_distill"] = _f(r"KL distillation:\s+([\d.]+)")
    d["t_final"] = _f(r"Edge final infer:\s+([\d.]+)")
    d["t_compute"] = _f(r"Total compute:\s+([\d.]+)")
    d["download_kb"] = _f(r"Download \(text_encoder\):\s+([\d.]+)")
    return d


def load_ams_logs(log_dir):
    data = {}
    for fname in os.listdir(log_dir):
        if not fname.endswith(".log"):
            continue
        ds = fname.replace(".log", "")
        data[ds] = parse_ams_log(os.path.join(log_dir, fname))
    return data


# ─── parse CAPO latency CSV ───

def load_capo_latency(csv_path):
    """Return {dataset_name: {field: value}}."""
    data = {}
    name_to_id = {v: k for k, v in DS_DISPLAY.items()}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            ds_name = row["Dataset"]
            ds_id = name_to_id.get(ds_name, ds_name)
            if ds_id in ("OOD Avg", "FGVC Avg", "Overall"):
                continue
            data[ds_id] = {k: float(v.replace("x", "")) if v.replace("x", "").replace(".", "").isdigit()
                           else v for k, v in row.items()}
    return data


# ─── estimate OOM datasets ───

def estimate_oom(ams_data, capo_data):
    """Use AMS/CAPO ratio from reference datasets to estimate OOM datasets."""
    ratios = {"cloud": [], "distill": [], "final": []}
    for ds in REFERENCE_DATASETS:
        if ds not in ams_data or ams_data[ds].get("oom"):
            continue
        ams = ams_data[ds]
        capo = capo_data[ds]
        capo_cloud = float(capo["CloudPromptOpt(ms)"])
        capo_distill = float(capo["CloudDistill(ms)"])
        capo_final = float(capo["EdgeFinal(ms)"])
        if capo_cloud > 0:
            ratios["cloud"].append(ams["t_cloud"] / capo_cloud)
        if capo_distill > 0:
            ratios["distill"].append(ams["t_distill"] / capo_distill)
        if capo_final > 0:
            ratios["final"].append(ams["t_final"] / capo_final)

    r_cloud = np.mean(ratios["cloud"])
    r_distill = np.mean(ratios["distill"])
    r_final = np.mean(ratios["final"])

    print(f"  AMS/CAPO ratio from {REFERENCE_DATASETS}:")
    print(f"    Cloud:   {r_cloud:.4f}  (AMS cloud infer / CAPO cloud prompt_opt)")
    print(f"    Distill: {r_distill:.4f}  (AMS KL distill / CAPO edge distill)")
    print(f"    Final:   {r_final:.4f}  (AMS edge final / CAPO edge final)")

    for ds in OOM_DATASETS:
        if ds not in capo_data:
            continue
        capo = capo_data[ds]
        est = {
            "oom": True,
            "estimated": True,
            "t_cloud": r_cloud * float(capo["CloudPromptOpt(ms)"]),
            "t_distill": r_distill * float(capo["CloudDistill(ms)"]),
            "t_final": r_final * float(capo["EdgeFinal(ms)"]),
            "download_kb": AMS_DOWNLOAD_KB,
            "acc": None,
        }
        est["t_compute"] = est["t_cloud"] + est["t_distill"] + est["t_final"]
        ams_data[ds] = est
        print(f"  Estimated {ds}: cloud={est['t_cloud']:.1f}ms  "
              f"distill={est['t_distill']:.1f}ms  "
              f"final={est['t_final']:.1f}ms  "
              f"total={est['t_compute']:.1f}ms")

    return ams_data


# ─── comm helper ───

def comm_ms(kb, bw_mbps):
    return kb * 1024 * 8 / (bw_mbps * 1e3)


# ─── build combined table ───

def build_table(ams_data, capo_data):
    rows = []
    for ds in ALL_DS:
        if ds not in ams_data:
            continue
        ams = ams_data[ds]
        capo = capo_data.get(ds, {})
        is_est = ams.get("estimated", False)

        up_kb = BITSTREAM_KB.get(ds, 0)
        ams_dn_kb = AMS_DOWNLOAD_KB
        ams_tot_kb = up_kb + ams_dn_kb
        capo_dn_kb = CAPO_DOWNLOAD_KB
        capo_tot_kb = up_kb + capo_dn_kb

        enc_ms, dec_ms = ENC_DEC_MS.get(ds, (0, 0))

        ams_compute = enc_ms + dec_ms + ams["t_cloud"] + ams["t_distill"] + ams["t_final"]
        capo_compute = float(capo.get("TotalCompute(ms)", 0))

        row = {
            "ds": ds,
            "name": DS_DISPLAY.get(ds, ds),
            "group": "OOD" if ds in OOD else "FGVC",
            "estimated": is_est,
            "ams_acc": ams.get("acc"),
            "up_kb": up_kb,
            "ams_dn_kb": ams_dn_kb,
            "ams_tot_kb": ams_tot_kb,
            "capo_dn_kb": capo_dn_kb,
            "capo_tot_kb": capo_tot_kb,
            "enc_ms": enc_ms,
            "dec_ms": dec_ms,
            "ams_cloud": ams["t_cloud"],
            "ams_distill": ams["t_distill"],
            "ams_final": ams["t_final"],
            "ams_compute": ams_compute,
            "capo_compute": capo_compute,
        }
        rows.append(row)
    return rows


def print_and_save(rows, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    # ── Text table ──
    W = 145
    sep = "=" * W
    lines = []

    def P(s=""):
        lines.append(s)
        print(s)

    P(f"\n{sep}")
    P("  AMS End-to-End System Evaluation — Summary")
    P(f"  Download per sample: text_encoder = {AMS_DOWNLOAD_KB:.0f} KB ({AMS_DOWNLOAD_KB/1024:.2f} MB)")
    P(f"  vs CAPO download: prompt = {CAPO_DOWNLOAD_KB} KB → ratio = {AMS_DOWNLOAD_KB/CAPO_DOWNLOAD_KB:.0f}×")
    P(sep)

    # ── Part A: Latency ──
    P(f"\n  Part A: AMS Latency Breakdown (all samples uplinked)")
    P(f"  {'─'*140}")
    h = (f"  {'Dataset':<12s} │{'Upload':>7s} {'Down':>8s} {'Total':>9s}│"
         f"{'ImgEnc':>7s} {'ImgDec':>7s} {'Cloud':>7s} {'Distill':>8s}"
         f" {'Final':>6s}│{'Compute':>8s}│"
         f" {'@1G':>8s} {'@100M':>9s} {'@50M':>9s}"
         f" {'@10M':>9s} {'@5M':>9s} {'@1M':>10s}│{'Note':>5s}")
    P(h)
    P(f"  {'':─<12s}─┼{'':─>7s}─{'':─>8s}─{'':─>9s}┼"
      f"{'':─>7s}─{'':─>7s}─{'':─>7s}─{'':─>8s}"
      f"─{'':─>6s}┼{'':─>8s}┼"
      f"─{'':─>8s}─{'':─>9s}─{'':─>9s}"
      f"─{'':─>9s}─{'':─>9s}─{'':─>10s}┼{'':─>5s}")

    for group_name, group_ds in [("OOD", OOD), ("FGVC", FEWSHOT)]:
        group_rows = [r for r in rows if r["ds"] in group_ds]
        for r in group_rows:
            bw_vals = [r["ams_compute"] + comm_ms(r["ams_tot_kb"], bw) for bw in BANDWIDTHS]
            note = "*est" if r["estimated"] else ""
            P(f"  {r['name']:<12s} │{r['up_kb']:7.2f} {r['ams_dn_kb']:8.0f} {r['ams_tot_kb']:9.0f}│"
              f"{r['enc_ms']:7.0f} {r['dec_ms']:7.0f} {r['ams_cloud']:7.1f} {r['ams_distill']:8.1f}"
              f" {r['ams_final']:6.1f}│{r['ams_compute']:8.1f}│"
              f" {bw_vals[0]:8.1f} {bw_vals[1]:9.1f} {bw_vals[2]:9.1f}"
              f" {bw_vals[3]:9.1f} {bw_vals[4]:9.1f} {bw_vals[5]:10.1f}│{note:>5s}")
        # group average
        if group_rows:
            avg = {k: np.mean([r[k] for r in group_rows])
                   for k in ["up_kb", "ams_dn_kb", "ams_tot_kb", "enc_ms", "dec_ms",
                              "ams_cloud", "ams_distill", "ams_final", "ams_compute"]}
            bw_vals = [avg["ams_compute"] + comm_ms(avg["ams_tot_kb"], bw) for bw in BANDWIDTHS]
            P(f"  {'▸'+group_name+' Avg':<12s} │{avg['up_kb']:7.2f} {avg['ams_dn_kb']:8.0f} {avg['ams_tot_kb']:9.0f}│"
              f"{avg['enc_ms']:7.0f} {avg['dec_ms']:7.0f} {avg['ams_cloud']:7.1f} {avg['ams_distill']:8.1f}"
              f" {avg['ams_final']:6.1f}│{avg['ams_compute']:8.1f}│"
              f" {bw_vals[0]:8.1f} {bw_vals[1]:9.1f} {bw_vals[2]:9.1f}"
              f" {bw_vals[3]:9.1f} {bw_vals[4]:9.1f} {bw_vals[5]:10.1f}│")
        if group_name == "OOD":
            P(f"  {'':─<12s}─┼{'':─>25s}┼{'':─>38s}┼{'':─>8s}┼{'':─>60s}┼{'':─>5s}")

    # overall average
    avg_all = {k: np.mean([r[k] for r in rows])
               for k in ["up_kb", "ams_dn_kb", "ams_tot_kb", "enc_ms", "dec_ms",
                          "ams_cloud", "ams_distill", "ams_final", "ams_compute"]}
    bw_all = [avg_all["ams_compute"] + comm_ms(avg_all["ams_tot_kb"], bw) for bw in BANDWIDTHS]
    P(f"  {'':─<12s}─┼{'':─>25s}┼{'':─>38s}┼{'':─>8s}┼{'':─>60s}┼{'':─>5s}")
    P(f"  {'▸Overall':<12s} │{avg_all['up_kb']:7.2f} {avg_all['ams_dn_kb']:8.0f} {avg_all['ams_tot_kb']:9.0f}│"
      f"{avg_all['enc_ms']:7.0f} {avg_all['dec_ms']:7.0f} {avg_all['ams_cloud']:7.1f} {avg_all['ams_distill']:8.1f}"
      f" {avg_all['ams_final']:6.1f}│{avg_all['ams_compute']:8.1f}│"
      f" {bw_all[0]:8.1f} {bw_all[1]:9.1f} {bw_all[2]:9.1f}"
      f" {bw_all[3]:9.1f} {bw_all[4]:9.1f} {bw_all[5]:10.1f}│")

    P(f"\n  * Upload = compressed image bitstream (QP1, same as CAPO)")
    P(f"  * Download = text_encoder params = {AMS_DOWNLOAD_KB:.0f} KB ({AMS_DOWNLOAD_KB/1024:.2f} MB)")
    P(f"  * *est = estimated from AMS/CAPO ratio (OOM on 24GB GPU)")

    # ── Part B: AMS vs CAPO comparison ──
    P(f"\n  Part B: AMS vs CAPO — Compute + Communication Comparison")
    P(f"  {'─'*120}")
    P(f"  {'Dataset':<12s} │ {'AMS Comp':>9s} {'CAPO Comp':>10s} {'Ratio':>6s}"
      f" │ {'AMS Comm':>9s} {'CAPO Comm':>10s} {'Ratio':>8s}"
      f" │ {'AMS@100M':>10s} {'CAPO@100M':>10s} {'Ratio':>6s}"
      f" │ {'AMS@10M':>10s} {'CAPO@10M':>10s} {'Ratio':>6s} │")
    P(f"  {'':─<12s}─┼─{'':─>9s}─{'':─>10s}─{'':─>6s}"
      f"─┼─{'':─>9s}─{'':─>10s}─{'':─>8s}"
      f"─┼─{'':─>10s}─{'':─>10s}─{'':─>6s}"
      f"─┼─{'':─>10s}─{'':─>10s}─{'':─>6s}─┤")

    for r in rows:
        ams_comm_100 = comm_ms(r["ams_tot_kb"], 100)
        capo_comm_100 = comm_ms(r["capo_tot_kb"], 100)
        ams_total_100 = r["ams_compute"] + ams_comm_100
        capo_total_100 = r["capo_compute"] + capo_comm_100
        ams_comm_10 = comm_ms(r["ams_tot_kb"], 10)
        capo_comm_10 = comm_ms(r["capo_tot_kb"], 10)
        ams_total_10 = r["ams_compute"] + ams_comm_10
        capo_total_10 = r["capo_compute"] + capo_comm_10

        comp_ratio = r["ams_compute"] / r["capo_compute"] if r["capo_compute"] > 0 else 0
        comm_ratio = r["ams_tot_kb"] / r["capo_tot_kb"] if r["capo_tot_kb"] > 0 else 0
        total_100_ratio = ams_total_100 / capo_total_100 if capo_total_100 > 0 else 0
        total_10_ratio = ams_total_10 / capo_total_10 if capo_total_10 > 0 else 0

        mark = "*" if r["estimated"] else " "
        P(f"  {r['name']:<12s}{mark}│ {r['ams_compute']:9.1f} {r['capo_compute']:10.1f} {comp_ratio:5.2f}×"
          f" │ {r['ams_tot_kb']:8.0f}K {r['capo_tot_kb']:9.2f}K {comm_ratio:6.0f}×"
          f" │ {ams_total_100:10.1f} {capo_total_100:10.1f} {total_100_ratio:5.1f}×"
          f" │ {ams_total_10:10.1f} {capo_total_10:10.1f} {total_10_ratio:5.1f}×"
          f" │")

    P(f"\n  Units: ms for time, KB for comm. @100M/@10M = total sync latency at that bandwidth.")
    P(f"  * = estimated (OOM)")

    P(f"\n{sep}\n")

    # ── Save text ──
    txt_path = os.path.join(out_dir, "ams_summary_tables.txt")
    with open(txt_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  Saved text: {txt_path}")

    # ── Save CSV: AMS latency ──
    csv_lat = os.path.join(out_dir, "ams_latency.csv")
    with open(csv_lat, "w", newline="") as f:
        w = csv.writer(f)
        bw_cols = [f"Sync@{bw}Mbps(ms)" for bw in BANDWIDTHS]
        w.writerow(["Group", "Dataset", "Estimated",
                    "Upload(KB)", "Download(KB)", "Total(KB)",
                    "ImgEncode(ms)", "ImgDecode(ms)",
                    "CloudInfer(ms)", "KLDistill(ms)", "EdgeFinal(ms)",
                    "TotalCompute(ms)"] + bw_cols)
        for r in rows:
            bw_vals = [round(r["ams_compute"] + comm_ms(r["ams_tot_kb"], bw), 2) for bw in BANDWIDTHS]
            w.writerow([
                r["group"], r["name"], "Y" if r["estimated"] else "N",
                round(r["up_kb"], 2), round(r["ams_dn_kb"], 0), round(r["ams_tot_kb"], 0),
                round(r["enc_ms"], 1), round(r["dec_ms"], 1),
                round(r["ams_cloud"], 2), round(r["ams_distill"], 2),
                round(r["ams_final"], 2), round(r["ams_compute"], 2),
            ] + bw_vals)
        # averages
        for label, subset in [("OOD Avg", [r for r in rows if r["group"] == "OOD"]),
                               ("FGVC Avg", [r for r in rows if r["group"] == "FGVC"]),
                               ("Overall", rows)]:
            avg = {k: np.mean([r[k] for r in subset])
                   for k in ["up_kb", "ams_dn_kb", "ams_tot_kb", "enc_ms", "dec_ms",
                              "ams_cloud", "ams_distill", "ams_final", "ams_compute"]}
            bw_vals = [round(avg["ams_compute"] + comm_ms(avg["ams_tot_kb"], bw), 2) for bw in BANDWIDTHS]
            w.writerow([
                label, label, "",
                round(avg["up_kb"], 2), round(avg["ams_dn_kb"], 0), round(avg["ams_tot_kb"], 0),
                round(avg["enc_ms"], 1), round(avg["dec_ms"], 1),
                round(avg["ams_cloud"], 2), round(avg["ams_distill"], 2),
                round(avg["ams_final"], 2), round(avg["ams_compute"], 2),
            ] + bw_vals)
    print(f"  Saved CSV:  {csv_lat}")

    # ── Save CSV: AMS vs CAPO ──
    csv_cmp = os.path.join(out_dir, "ams_vs_capo.csv")
    with open(csv_cmp, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Group", "Dataset", "Estimated",
                    "AMS_Compute(ms)", "CAPO_Compute(ms)", "ComputeRatio",
                    "AMS_Comm(KB)", "CAPO_Comm(KB)", "CommRatio",
                    "AMS_Total@100M(ms)", "CAPO_Total@100M(ms)", "TotalRatio@100M",
                    "AMS_Total@10M(ms)", "CAPO_Total@10M(ms)", "TotalRatio@10M"])
        for r in rows:
            ams_c100 = r["ams_compute"] + comm_ms(r["ams_tot_kb"], 100)
            capo_c100 = r["capo_compute"] + comm_ms(r["capo_tot_kb"], 100)
            ams_c10 = r["ams_compute"] + comm_ms(r["ams_tot_kb"], 10)
            capo_c10 = r["capo_compute"] + comm_ms(r["capo_tot_kb"], 10)
            w.writerow([
                r["group"], r["name"], "Y" if r["estimated"] else "N",
                round(r["ams_compute"], 2), round(r["capo_compute"], 2),
                round(r["ams_compute"] / r["capo_compute"], 3) if r["capo_compute"] > 0 else "",
                round(r["ams_tot_kb"], 2), round(r["capo_tot_kb"], 2),
                round(r["ams_tot_kb"] / r["capo_tot_kb"], 0) if r["capo_tot_kb"] > 0 else "",
                round(ams_c100, 2), round(capo_c100, 2),
                round(ams_c100 / capo_c100, 2) if capo_c100 > 0 else "",
                round(ams_c10, 2), round(capo_c10, 2),
                round(ams_c10 / capo_c10, 2) if capo_c10 > 0 else "",
            ])
    print(f"  Saved CSV:  {csv_cmp}")


# ─── main ───

def main():
    base = os.path.dirname(os.path.abspath(__file__))
    ams_log_dir = os.path.join(base, "logs", "ams_bench")
    capo_csv = os.path.join(base, "results", "e2e_bench", "summary_latency.csv")
    out_dir = os.path.join(base, "results", "ams_bench")

    print("Loading AMS logs...")
    ams_data = load_ams_logs(ams_log_dir)

    print("Loading CAPO latency CSV...")
    capo_data = load_capo_latency(capo_csv)

    print("\nOOM datasets:", [ds for ds in ALL_DS if ds in ams_data and ams_data[ds].get("oom")])
    print("Successful:",  [ds for ds in ALL_DS if ds in ams_data and not ams_data[ds].get("oom")])

    print("\nEstimating OOM datasets from AMS/CAPO ratio...")
    ams_data = estimate_oom(ams_data, capo_data)

    print("\nBuilding tables...")
    rows = build_table(ams_data, capo_data)
    print_and_save(rows, out_dir)


if __name__ == "__main__":
    main()
