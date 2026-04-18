#!/usr/bin/env python3
# coding=utf-8

import argparse
import csv
import os
from collections import defaultdict

import numpy as np


def comm_ms(total_kb, bw_mbps):
    return total_kb * 1024.0 * 8.0 / (bw_mbps * 1e3)


def load_rows(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    out = []
    for r in rows:
        out.append(
            {
                "task": r["task"],
                "idx": int(r["idx"]),
                "upload_kb": float(r["upload_kb"]),
                "download_kb": float(r["download_kb"]),
                "t_teacher_ms": float(r["t_teacher_ms"]),
                "t_adapt_ms": float(r["t_adapt_ms"]),
                "t_final_ms": float(r["t_final_ms"]),
            }
        )
    return out


def summarize(rows):
    arr = defaultdict(list)
    for r in rows:
        for k, v in r.items():
            if k in ("task", "idx"):
                continue
            arr[k].append(v)
    mean = {k: float(np.mean(v)) for k, v in arr.items()}
    std = {k: float(np.std(v)) for k, v in arr.items()}
    mean["t_total_ms"] = mean["t_teacher_ms"] + mean["t_adapt_ms"] + mean["t_final_ms"]
    return mean, std


def print_summary(i2t, t2i, bandwidths):
    print("=" * 100)
    print("AMS Retrieval E2E Breakdown")
    print("=" * 100)
    print(
        f"{'Task':<6} {'UploadKB':>10} {'DownKB':>9} {'Teacher':>10} "
        f"{'Adapt':>10} {'Final':>10} {'Total':>10}"
    )
    print("-" * 100)
    for name, m in [("i2t", i2t[0]), ("t2i", t2i[0])]:
        print(
            f"{name:<6} {m['upload_kb']:10.3f} {m['download_kb']:9.3f} "
            f"{m['t_teacher_ms']:10.2f} {m['t_adapt_ms']:10.2f} "
            f"{m['t_final_ms']:10.2f} {m['t_total_ms']:10.2f}"
        )
    print("=" * 100)
    for name, m in [("i2t", i2t[0]), ("t2i", t2i[0])]:
        total_kb = m["upload_kb"] + m["download_kb"]
        parts = []
        for bw in bandwidths:
            total_ms = m["t_total_ms"] + comm_ms(total_kb, bw)
            parts.append(f"{bw}Mbps:{total_ms:.2f}ms")
        print(f"{name}: " + " | ".join(parts))
    print("=" * 100)


def save_csv(path, i2t, t2i, bandwidths):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "task",
                "upload_kb",
                "download_kb",
                "t_teacher_ms",
                "t_adapt_ms",
                "t_final_ms",
                "t_total_ms",
            ]
            + [f"total_ms@{bw}Mbps" for bw in bandwidths]
        )
        for name, m in [("i2t", i2t[0]), ("t2i", t2i[0])]:
            total_kb = m["upload_kb"] + m["download_kb"]
            vals = [m["t_total_ms"] + comm_ms(total_kb, bw) for bw in bandwidths]
            w.writerow(
                [
                    name,
                    round(m["upload_kb"], 4),
                    round(m["download_kb"], 4),
                    round(m["t_teacher_ms"], 4),
                    round(m["t_adapt_ms"], 4),
                    round(m["t_final_ms"], 4),
                    round(m["t_total_ms"], 4),
                ]
                + [round(v, 4) for v in vals]
            )


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--i2t-csv", required=True)
    p.add_argument("--t2i-csv", required=True)
    p.add_argument("--out-csv", required=True)
    p.add_argument("--bandwidths", type=int, nargs="+", default=[1000, 100, 50, 10, 5, 1])
    return p.parse_args()


def main():
    args = parse_args()
    i2t = summarize(load_rows(args.i2t_csv))
    t2i = summarize(load_rows(args.t2i_csv))
    print_summary(i2t, t2i, args.bandwidths)
    save_csv(args.out_csv, i2t, t2i, args.bandwidths)
    print(f"Summary CSV saved: {args.out_csv}")


if __name__ == "__main__":
    main()
