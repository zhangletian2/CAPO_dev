#!/usr/bin/env python3
"""
gen_e2e_tables.py — 合并 e2e_bench + image_enc_dec_times 生成论文表格

数据源:
  1. logs/e2e_bench/*.log      → Cloud TTA / Edge 推理时间
  2. image_enc_dec_times.csv   → 图像编码/解码时间
  3. draw_compress_prompt.py   → prompt 压缩后 3.99 KB (FP32+zlib, n_ctx=4)
  4. BITSTREAM_KB 内置表       → 每张图像压缩码流大小

输出:
  Table 1: Communication cost
  Table 2: Full latency breakdown (含 enc/dec)
  Table 3: Bandwidth-dependent total latency
  Table 4: Sync vs Async accuracy × uplink%
"""

import csv
import os
import re
import sys
import numpy as np

# ─── constants ───

PROMPT_DOWNLOAD_KB = 3.99  # FP32 + zlib, n_ctx=4
AMS_KB = 9464.58

BITSTREAM_KB = {
    "new": {
        "I": 7.88, "A": 6.40, "R": 6.63, "K": 13.13, "V": 7.13,
        "flower102": 5.68, "dtd": 8.85, "eurosat": 0.13, "cars": 8.03,
        "pets": 4.76, "food101": 4.24, "ucf101": 1.80, "sun397": 25.23,
        "aircraft": 14.60, "caltech101": 3.02,
    },
}

BANDWIDTHS = [1000, 100, 50, 10, 5, 1]

OOD = ["I", "A", "R", "K", "V"]
FEWSHOT = ["flower102", "dtd", "eurosat", "cars", "pets",
           "food101", "ucf101", "sun397", "aircraft", "caltech101"]
ALL_DS = OOD + FEWSHOT

DS_DISPLAY = {
    "I": "ImageNet", "A": "ImageNet-A", "R": "ImageNet-R",
    "K": "ImageNet-K", "V": "ImageNet-V2",
}

# ─── parse enc/dec CSV ───

def parse_enc_dec_csv(path):
    """Return {dataset_id: (encode_ms, decode_ms)}."""
    id_map = {
        "I (imagenet)": "I", "K (sketch)": "K", "R (rendition)": "R",
        "V (v2)": "V", "A (adversarial)": "A",
        "food101": "food101", "sun397": "sun397", "cars": "cars",
        "eurosat": "eurosat", "ucf101": "ucf101", "pets": "pets",
        "aircraft": "aircraft", "caltech101": "caltech101",
        "flower102": "flower102", "dtd": "dtd",
    }
    result = {}
    with open(path, encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        next(reader)  # header
        for row in reader:
            if len(row) < 6:
                continue
            name = row[0].strip().replace("\xa0", " ")
            ds_id = id_map.get(name, name)
            enc = float(row[3].replace("ms", "").replace(",", "").strip())
            dec = float(row[4].replace("ms", "").replace(",", "").strip())
            result[ds_id] = (enc, dec)
    return result


# ─── parse bench logs ───

def parse_bench_log(path):
    """Extract key metrics from one bench log file."""
    d = {}
    with open(path) as f:
        text = f.read()

    def _float(pattern):
        m = re.search(pattern, text)
        return float(m.group(1)) if m else 0.0

    d["sync_acc"] = _float(r"Sync  \(post-TTA\):\s+([\d.]+)%")
    d["async_acc"] = _float(r"Async \(pre-TTA\):\s+([\d.]+)%")
    d["t_entropy"] = _float(r"Edge entropy check:\s+([\d.]+)")
    d["t_cloud_tta"] = _float(r"Cloud TTA \(prompt opt\):\s+([\d.]+)")
    d["t_distill"] = _float(r"Cloud \(edge distill\):\s+([\d.]+)")
    d["t_cloud_total"] = _float(r"── Cloud total:\s+([\d.]+)")
    d["t_final"] = _float(r"Edge final inference:\s+([\d.]+)")
    d["t_compute"] = _float(r"── Total compute:\s+([\d.]+)")
    return d


def load_all_logs(log_dir):
    """Return {(ds, uplink%): metrics_dict}."""
    data = {}
    for fname in os.listdir(log_dir):
        if not fname.endswith(".log"):
            continue
        m = re.match(r"(.+)_up(\d+)\.log$", fname)
        if not m:
            continue
        ds, up = m.group(1), int(m.group(2))
        data[(ds, up)] = parse_bench_log(os.path.join(log_dir, fname))
    return data


def comm_ms(kb, bw_mbps):
    return kb * 1024 * 8 / (bw_mbps * 1e3)


# ─── table generators ───

def table_comm(enc_dec):
    """Table 1: Communication Cost per Uplinked Sample."""
    print("\n" + "=" * 75)
    print("  Table 1: Communication Cost per Uplinked Sample")
    print("=" * 75)
    print(f"  {'Dataset':<12s}  {'Upload(KB)':>10s}  {'Download(KB)':>12s}"
          f"  {'Total(KB)':>10s}  {'vs AMS':>8s}")
    print(f"  {'─'*12}  {'─'*10}  {'─'*12}  {'─'*10}  {'─'*8}")
    upload_vals, total_vals = [], []
    for ds in ALL_DS:
        up_kb = BITSTREAM_KB["new"].get(ds, 0)
        dn_kb = PROMPT_DOWNLOAD_KB
        tot = up_kb + dn_kb
        ratio = AMS_KB / tot if tot > 0 else 0
        upload_vals.append(up_kb)
        total_vals.append(tot)
        name = DS_DISPLAY.get(ds, ds)
        print(f"  {name:<12s}  {up_kb:10.2f}  {dn_kb:12.2f}"
              f"  {tot:10.2f}  {ratio:7.0f}×")
    print(f"  {'─'*12}  {'─'*10}  {'─'*12}  {'─'*10}  {'─'*8}")
    import numpy as np
    print(f"  {'Average':<12s}  {np.mean(upload_vals):10.2f}"
          f"  {PROMPT_DOWNLOAD_KB:12.2f}  {np.mean(total_vals):10.2f}"
          f"  {AMS_KB/np.mean(total_vals):7.0f}×")
    print(f"\n  * Upload = compressed image bitstream (QP1, Enc2Dec)")
    print(f"  * Download = prompt ctx (FP32 + zlib) = {PROMPT_DOWNLOAD_KB} KB")
    print(f"  * AMS baseline = {AMS_KB} KB/sample")


def table_latency(bench_data, enc_dec):
    """Table 2: Full Latency Breakdown per Uplinked Sample (100% uplink)."""
    print("\n" + "=" * 110)
    print("  Table 2: Full Latency Breakdown per Uplinked Sample (ms, uplink=100%)")
    print("  Pipeline: Edge[entropy→encode] → Network[upload] → Cloud[decode→TTA] → Network[download] → Edge[final]")
    print("=" * 110)
    header = (f"  {'Dataset':<12s}  {'Entropy':>8s}  {'ImgEnc':>7s}  "
              f"{'ImgDec':>7s}  {'CloudTTA':>9s}  {'Distill':>8s}  "
              f"{'Final':>6s}  {'Compute':>8s}  {'Sync@100M':>10s}  {'Async':>7s}")
    print(header)
    print(f"  {'─'*12}  {'─'*8}  {'─'*7}  {'─'*7}  {'─'*9}  {'─'*8}"
          f"  {'─'*6}  {'─'*8}  {'─'*10}  {'─'*7}")

    for ds in ALL_DS:
        key = (ds, 100)
        if key not in bench_data:
            continue
        b = bench_data[key]
        enc_ms, dec_ms = enc_dec.get(ds, (0, 0))
        up_kb = BITSTREAM_KB["new"].get(ds, 0)
        tot_kb = up_kb + PROMPT_DOWNLOAD_KB

        compute_full = (b["t_entropy"] + enc_ms + dec_ms
                        + b["t_cloud_total"] + b["t_final"])
        comm_100 = comm_ms(tot_kb, 100)
        sync_100 = compute_full + comm_100
        async_t = b["t_entropy"]

        name = DS_DISPLAY.get(ds, ds)
        print(f"  {name:<12s}  {b['t_entropy']:8.1f}  {enc_ms:7.0f}  "
              f"{dec_ms:7.0f}  {b['t_cloud_tta']:9.1f}  {b['t_distill']:8.1f}  "
              f"{b['t_final']:6.1f}  {compute_full:8.1f}  {sync_100:10.1f}  "
              f"{async_t:7.1f}")

    print(f"\n  * ImgEnc = Enc2Dec encoding (edge side)")
    print(f"  * ImgDec = Enc2Dec decoding (cloud side)")
    print(f"  * Sync@100M = total at 100 Mbps bandwidth")
    print(f"  * Async = edge-only latency (no cloud round-trip)")


def table_bandwidth(bench_data, enc_dec):
    """Table 3: Total Sync Latency under Different Bandwidths."""
    print("\n" + "=" * 90)
    print("  Table 3: Total Sync Latency under Different Bandwidths (ms, uplink=100%)")
    print("=" * 90)
    bw_hdrs = "  ".join(f"{bw:>7d}M" for bw in BANDWIDTHS)
    print(f"  {'Dataset':<12s}  {'Compute':>8s}  {bw_hdrs}")
    print(f"  {'─'*12}  {'─'*8}  " + "  ".join(["─" * 8] * len(BANDWIDTHS)))

    for ds in ALL_DS:
        key = (ds, 100)
        if key not in bench_data:
            continue
        b = bench_data[key]
        enc_ms, dec_ms = enc_dec.get(ds, (0, 0))
        up_kb = BITSTREAM_KB["new"].get(ds, 0)
        tot_kb = up_kb + PROMPT_DOWNLOAD_KB

        compute = (b["t_entropy"] + enc_ms + dec_ms
                   + b["t_cloud_total"] + b["t_final"])
        name = DS_DISPLAY.get(ds, ds)
        vals = []
        for bw in BANDWIDTHS:
            total = compute + comm_ms(tot_kb, bw)
            vals.append(f"{total:8.1f}")
        print(f"  {name:<12s}  {compute:8.1f}  {'  '.join(vals)}")

    print(f"\n  * Compute includes: edge entropy + img encode + img decode"
          f" + cloud TTA + edge final")
    print(f"  * Communication = (image bitstream + prompt 3.99KB)"
          f" × 8 / bandwidth")


def table_accuracy(bench_data):
    """Table 4: Sync vs Async Accuracy at Different Uplink Rates."""
    print("\n" + "=" * 100)
    print("  Table 4: Sync vs Async Accuracy (%) at Different Uplink Rates")
    print("=" * 100)
    uplinks = [25, 50, 75, 100]
    hdr = "  ".join(f"{'Sync':>6s}/{'Async':<6s}" for _ in uplinks)
    up_hdr = "  ".join(f"  up={u}%{'':>4s}" for u in uplinks)
    print(f"  {'Dataset':<12s}  {up_hdr}")
    print(f"  {'─'*12}  " + "  ".join(["─" * 13] * len(uplinks)))

    for ds in ALL_DS:
        name = DS_DISPLAY.get(ds, ds)
        parts = []
        for up in uplinks:
            key = (ds, up)
            if key in bench_data:
                b = bench_data[key]
                delta = b["sync_acc"] - b["async_acc"]
                parts.append(f"{b['sync_acc']:5.1f}/{b['async_acc']:<5.1f}"
                             f"({delta:+.1f})")
            else:
                parts.append(f"{'N/A':>13s}")
        print(f"  {name:<12s}  {'  '.join(parts)}")

    print(f"\n  * Sync = post-TTA prediction; Async = pre-TTA (edge-only)")
    print(f"  * Δ = sync − async; positive means TTA helps")


def table_capo_vs_ams():
    """Table 5: CAPO vs AMS Communication Latency."""
    import numpy as np
    avg_total = np.mean([BITSTREAM_KB["new"][ds] + PROMPT_DOWNLOAD_KB
                         for ds in ALL_DS])
    print("\n" + "=" * 70)
    print("  Table 5: Communication Latency — CAPO vs AMS (ms)")
    print(f"  CAPO avg: {avg_total:.2f} KB/sample   AMS: {AMS_KB:.0f} KB/sample")
    print("=" * 70)
    print(f"  {'Bandwidth':>12s}  {'CAPO':>10s}  {'AMS':>10s}  {'Speedup':>8s}")
    print(f"  {'─'*12}  {'─'*10}  {'─'*10}  {'─'*8}")
    for bw in BANDWIDTHS:
        c = comm_ms(avg_total, bw)
        a = comm_ms(AMS_KB, bw)
        print(f"  {bw:>8d} Mbps  {c:8.2f}ms  {a:8.0f}ms  {a/c:7.0f}×")


# ─── summary table ───

def _build_rows(bench_data, enc_dec):
    """Build per-dataset row dicts with all combined metrics."""
    rows = []
    for ds in ALL_DS:
        key100 = (ds, 100)
        if key100 not in bench_data:
            continue
        b = bench_data[key100]
        enc_ms, dec_ms = enc_dec.get(ds, (0, 0))
        up_kb = BITSTREAM_KB["new"].get(ds, 0)
        dn_kb = PROMPT_DOWNLOAD_KB
        tot_kb = up_kb + dn_kb

        cloud_proc = b["t_cloud_total"]  # prompt opt + distill on cloud
        compute = b["t_entropy"] + enc_ms + dec_ms + cloud_proc + b["t_final"]

        sync_accs, async_accs, deltas = [], [], []
        for up in [25, 50, 75, 100]:
            k = (ds, up)
            if k in bench_data:
                s = bench_data[k]["sync_acc"]
                a = bench_data[k]["async_acc"]
                sync_accs.append(s)
                async_accs.append(a)
                deltas.append(s - a)

        rows.append({
            "ds": ds,
            "name": DS_DISPLAY.get(ds, ds),
            "group": "OOD" if ds in OOD else "FGVC",
            "up_kb": up_kb, "dn_kb": dn_kb, "tot_kb": tot_kb,
            "t_entropy": b["t_entropy"],
            "t_enc": enc_ms, "t_dec": dec_ms,
            "t_cloud_tta": b["t_cloud_tta"],
            "t_distill": b["t_distill"],
            "t_cloud": cloud_proc,
            "t_final": b["t_final"],
            "t_compute": compute,
            "sync_accs": {up: bench_data[(ds, up)]["sync_acc"]
                         for up in [25, 50, 75, 100] if (ds, up) in bench_data},
            "async_accs": {up: bench_data[(ds, up)]["async_acc"]
                          for up in [25, 50, 75, 100] if (ds, up) in bench_data},
            "delta_avg": np.mean(deltas) if deltas else 0,
        })
    return rows


def _avg_row(rows, label):
    """Compute average of a list of row dicts."""
    if not rows:
        return None
    keys_f = ["up_kb", "dn_kb", "tot_kb", "t_entropy", "t_enc", "t_dec",
              "t_cloud_tta", "t_distill", "t_cloud", "t_final", "t_compute",
              "delta_avg"]
    avg = {k: np.mean([r[k] for r in rows]) for k in keys_f}
    avg["name"] = label
    avg["ds"] = label
    avg["group"] = label
    avg["sync_accs"] = {
        up: np.mean([r["sync_accs"][up] for r in rows if up in r["sync_accs"]])
        for up in [25, 50, 75, 100]
    }
    avg["async_accs"] = {
        up: np.mean([r["async_accs"][up] for r in rows if up in r["async_accs"]])
        for up in [25, 50, 75, 100]
    }
    return avg


def table_summary(bench_data, enc_dec):
    """Master summary table combining all metrics, with group averages."""
    rows = _build_rows(bench_data, enc_dec)
    ood_rows = [r for r in rows if r["group"] == "OOD"]
    fgvc_rows = [r for r in rows if r["group"] == "FGVC"]

    ood_avg = _avg_row(ood_rows, "OOD Avg")
    fgvc_avg = _avg_row(fgvc_rows, "FGVC Avg")
    all_avg = _avg_row(rows, "Overall")

    W = 140
    print("\n" + "=" * W)
    print("  SUMMARY TABLE: End-to-End System Evaluation per Uplinked Sample")
    print("  Prompt: 4 tokens × 512d, FP32+zlib = 3.99 KB  |  Image codec: Enc2Dec QP1")
    print("=" * W)

    # ── Part A: Communication + Latency ──
    print(f"\n  Part A: Communication Volume & Latency Breakdown (uplink=100%)")
    print(f"  {'─'*136}")
    h = (f"  {'Dataset':<12s} │{'Upload':>7s} {'Down':>5s} {'Total':>6s}│"
         f"{'Entropy':>8s} {'Encode':>7s} {'Decode':>7s} {'PromOpt':>8s}"
         f" {'Distill':>8s} {'Final':>6s}│{'Compute':>8s}│"
         f" {'@1G':>6s} {'@100M':>7s} {'@50M':>7s} {'@10M':>7s}"
         f" {'@5M':>7s} {'@1M':>7s}│{'Async':>6s}")
    print(h)
    print(f"  {'':─<12s}─┼{'':─>7s}─{'':─>5s}─{'':─>6s}┼"
          f"{'':─>8s}─{'':─>7s}─{'':─>7s}─{'':─>8s}"
          f"─{'':─>8s}─{'':─>6s}┼{'':─>8s}┼"
          f"─{'':─>6s}─{'':─>7s}─{'':─>7s}─{'':─>7s}"
          f"─{'':─>7s}─{'':─>7s}┼{'':─>6s}")

    def _print_row(r, bold=False):
        nm = r["name"]
        if bold:
            nm = f"▸{nm}"
        bw_vals = []
        for bw in BANDWIDTHS:
            bw_vals.append(r["t_compute"] + comm_ms(r["tot_kb"], bw))
        print(f"  {nm:<12s} │{r['up_kb']:7.2f} {r['dn_kb']:5.2f} {r['tot_kb']:6.2f}│"
              f"{r['t_entropy']:8.1f} {r['t_enc']:7.0f} {r['t_dec']:7.0f}"
              f" {r['t_cloud_tta']:8.1f} {r['t_distill']:8.1f} {r['t_final']:6.1f}│"
              f"{r['t_compute']:8.1f}│"
              f" {bw_vals[0]:6.1f} {bw_vals[1]:7.1f} {bw_vals[2]:7.1f}"
              f" {bw_vals[3]:7.1f} {bw_vals[4]:7.1f} {bw_vals[5]:7.1f}│"
              f"{r['t_entropy']:6.1f}")

    for r in ood_rows:
        _print_row(r)
    if ood_avg:
        _print_row(ood_avg, bold=True)
    print(f"  {'':─<12s}─┼{'':─>19s}┼{'':─>49s}┼{'':─>8s}┼{'':─>44s}┼{'':─>6s}")
    for r in fgvc_rows:
        _print_row(r)
    if fgvc_avg:
        _print_row(fgvc_avg, bold=True)
    print(f"  {'':─<12s}─┼{'':─>19s}┼{'':─>49s}┼{'':─>8s}┼{'':─>44s}┼{'':─>6s}")
    if all_avg:
        _print_row(all_avg, bold=True)

    print(f"\n  Units: KB for comm, ms for time. Sync@BW = Compute + Comm latency.")
    print(f"  Compute = Entropy + Encode + Decode + PromptOpt + Distill + Final")
    print(f"  Async = edge entropy check only (no cloud round-trip)")

    # ── Part B: Accuracy ──
    print(f"\n  Part B: Sync / Async Accuracy (%) & Δ at Different Uplink Rates")
    print(f"  {'─'*120}")
    print(f"  {'Dataset':<12s} │ {'up=25%':^18s} │ {'up=50%':^18s}"
          f" │ {'up=75%':^18s} │ {'up=100%':^18s} │ {'Avg Δ':>6s}")
    print(f"  {'':─<12s}─┼─{'':─>18s}─┼─{'':─>18s}"
          f"─┼─{'':─>18s}─┼─{'':─>18s}─┼─{'':─>6s}")

    def _print_acc_row(r, bold=False):
        nm = f"▸{r['name']}" if bold else r["name"]
        parts = []
        for up in [25, 50, 75, 100]:
            s = r["sync_accs"].get(up, 0)
            a = r["async_accs"].get(up, 0)
            d = s - a
            parts.append(f"{s:5.1f}/{a:5.1f}({d:+4.1f})")
        print(f"  {nm:<12s} │ {'│'.join(f' {p} ' for p in parts)}│ {r['delta_avg']:+5.1f}")

    for r in ood_rows:
        _print_acc_row(r)
    if ood_avg:
        _print_acc_row(ood_avg, bold=True)
    print(f"  {'':─<12s}─┼─{'':─>18s}─┼─{'':─>18s}"
          f"─┼─{'':─>18s}─┼─{'':─>18s}─┼─{'':─>6s}")
    for r in fgvc_rows:
        _print_acc_row(r)
    if fgvc_avg:
        _print_acc_row(fgvc_avg, bold=True)
    print(f"  {'':─<12s}─┼─{'':─>18s}─┼─{'':─>18s}"
          f"─┼─{'':─>18s}─┼─{'':─>18s}─┼─{'':─>6s}")
    if all_avg:
        _print_acc_row(all_avg, bold=True)

    # ── Part C: vs AMS ──
    avg_kb = all_avg["tot_kb"] if all_avg else 11.82
    print(f"\n  Part C: Communication Cost — CAPO ({avg_kb:.2f} KB) vs AMS ({AMS_KB:.0f} KB)")
    print(f"  {'─'*60}")
    print(f"  {'Bandwidth':>12s}  {'CAPO':>10s}  {'AMS':>10s}  {'Speedup':>8s}")
    print(f"  {'─'*12}  {'─'*10}  {'─'*10}  {'─'*8}")
    for bw in BANDWIDTHS:
        c = comm_ms(avg_kb, bw)
        a = comm_ms(AMS_KB, bw)
        print(f"  {bw:>8d} Mbps  {c:8.2f}ms  {a:8.0f}ms  {a/c:7.0f}×")


# ─── CSV export ───

def save_csvs(bench_data, enc_dec, out_dir):
    rows = _build_rows(bench_data, enc_dec)
    ood_rows = [r for r in rows if r["group"] == "OOD"]
    fgvc_rows = [r for r in rows if r["group"] == "FGVC"]
    ood_avg = _avg_row(ood_rows, "OOD Avg")
    fgvc_avg = _avg_row(fgvc_rows, "FGVC Avg")
    all_avg = _avg_row(rows, "Overall")

    all_rows = (ood_rows + ([ood_avg] if ood_avg else [])
                + fgvc_rows + ([fgvc_avg] if fgvc_avg else [])
                + ([all_avg] if all_avg else []))

    os.makedirs(out_dir, exist_ok=True)

    # ── Part A: Latency ──
    pa = os.path.join(out_dir, "summary_latency.csv")
    with open(pa, "w", newline="") as f:
        w = csv.writer(f)
        bw_cols = [f"Sync@{bw}Mbps(ms)" for bw in BANDWIDTHS]
        w.writerow(["Group", "Dataset",
                    "Upload(KB)", "Download(KB)", "Total(KB)", "vsAMS",
                    "EdgeEntropy(ms)", "ImgEncode(ms)", "ImgDecode(ms)",
                    "CloudPromptOpt(ms)", "CloudDistill(ms)", "EdgeFinal(ms)",
                    "TotalCompute(ms)"] + bw_cols + ["Async(ms)"])
        for r in all_rows:
            ratio = AMS_KB / r["tot_kb"] if r["tot_kb"] > 0 else 0
            bw_vals = [round(r["t_compute"] + comm_ms(r["tot_kb"], bw), 2)
                       for bw in BANDWIDTHS]
            w.writerow([
                r["group"], r["name"],
                round(r["up_kb"], 2), round(r["dn_kb"], 2),
                round(r["tot_kb"], 2), f"{ratio:.0f}x",
                round(r["t_entropy"], 2), round(r["t_enc"], 1),
                round(r["t_dec"], 1), round(r["t_cloud_tta"], 2),
                round(r["t_distill"], 2), round(r["t_final"], 2),
                round(r["t_compute"], 2),
            ] + bw_vals + [round(r["t_entropy"], 2)])
    print(f"  Saved: {pa}")

    # ── Part B: Accuracy ──
    pb = os.path.join(out_dir, "summary_accuracy.csv")
    with open(pb, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Group", "Dataset",
                    "Sync@25%", "Async@25%", "Delta@25%",
                    "Sync@50%", "Async@50%", "Delta@50%",
                    "Sync@75%", "Async@75%", "Delta@75%",
                    "Sync@100%", "Async@100%", "Delta@100%",
                    "AvgDelta"])
        for r in all_rows:
            row = [r["group"], r["name"]]
            for up in [25, 50, 75, 100]:
                s = r["sync_accs"].get(up, "")
                a = r["async_accs"].get(up, "")
                d = round(s - a, 2) if s != "" and a != "" else ""
                row += [round(s, 2) if s != "" else "",
                        round(a, 2) if a != "" else "", d]
            row.append(round(r["delta_avg"], 2))
            w.writerow(row)
    print(f"  Saved: {pb}")

    # ── Part C: CAPO vs AMS ──
    pc = os.path.join(out_dir, "summary_capo_vs_ams.csv")
    avg_kb = all_avg["tot_kb"] if all_avg else 11.82
    with open(pc, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Bandwidth(Mbps)",
                    f"CAPO({avg_kb:.2f}KB)(ms)",
                    f"AMS({AMS_KB:.0f}KB)(ms)", "Speedup"])
        for bw in BANDWIDTHS:
            c = comm_ms(avg_kb, bw)
            a = comm_ms(AMS_KB, bw)
            w.writerow([bw, round(c, 2), round(a, 0), f"{a/c:.0f}x"])
    print(f"  Saved: {pc}")


# ─── main ───

def main():
    base = os.path.dirname(os.path.abspath(__file__))
    log_dir = os.path.join(base, "logs", "e2e_bench")
    enc_dec_csv = os.path.join(base, "image_enc_dec_times.csv")
    out_dir = os.path.join(base, "results", "e2e_bench")

    enc_dec = parse_enc_dec_csv(enc_dec_csv)
    bench_data = load_all_logs(log_dir)

    print("\n" + "█" * 75)
    print("  CAPO End-to-End System Evaluation — Paper Tables")
    print("█" * 75)

    table_summary(bench_data, enc_dec)
    save_csvs(bench_data, enc_dec, out_dir)

    print("\n" + "█" * 75)
    print("  Done.")
    print("█" * 75 + "\n")


if __name__ == "__main__":
    main()
