#!/usr/bin/env python3
# coding=utf-8
"""
Aggregate retrieval E2E benchmark CSVs into simple tables.
"""

import argparse
import csv
import os
from collections import defaultdict

import numpy as np


def comm_ms(total_kb, bw_mbps):
    return total_kb * 1024.0 * 8.0 / (bw_mbps * 1e3)


def load_csv(path):
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows


def to_float_rows(rows):
    out = []
    for r in rows:
        out.append(
            {
                "task": r["task"],
                "idx": int(r["idx"]),
                "upload_kb": float(r["upload_kb"]),
                "download_kb": float(r["download_kb"]),
                "t_edge_prep_ms": float(r["t_edge_prep_ms"]),
                "t_cloud_prompt_ms": float(r["t_cloud_prompt_ms"]),
                "t_edge_distill_ms": float(r["t_edge_distill_ms"]),
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
    mean["total_compute_ms"] = (
        mean["t_edge_prep_ms"]
        + mean["t_cloud_prompt_ms"]
        + mean["t_edge_distill_ms"]
        + mean["t_final_ms"]
    )
    return mean, std


def print_table(summary_by_task, bandwidths):
    print("=" * 96)
    print("Retrieval E2E Latency Breakdown (per sample)")
    print("=" * 96)
    print(
        f"{'Task':<6}  {'UploadKB':>9}  {'DownKB':>8}  {'EdgePrep':>9}  "
        f"{'CloudTTA':>9}  {'EdgeKL':>8}  {'Final':>8}  {'Compute':>9}"
    )
    print("-" * 96)
    for task, (m, s) in summary_by_task.items():
        print(
            f"{task:<6}  "
            f"{m['upload_kb']:9.3f}  {m['download_kb']:8.3f}  "
            f"{m['t_edge_prep_ms']:9.2f}  {m['t_cloud_prompt_ms']:9.2f}  "
            f"{m['t_edge_distill_ms']:8.2f}  {m['t_final_ms']:8.2f}  "
            f"{m['total_compute_ms']:9.2f}"
        )
    print("=" * 96)
    print("Bandwidth-dependent total latency:")
    for task, (m, _) in summary_by_task.items():
        total_kb = m["upload_kb"] + m["download_kb"]
        parts = []
        for bw in bandwidths:
            total_ms = m["total_compute_ms"] + comm_ms(total_kb, bw)
            parts.append(f"{bw}Mbps:{total_ms:.2f}ms")
        print(f"  {task}: " + " | ".join(parts))
    print("=" * 96)


def save_summary_csv(path, summary_by_task, bandwidths):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    bw_cols = [f"total_ms@{bw}Mbps" for bw in bandwidths]
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "task",
                "upload_kb",
                "download_kb",
                "t_edge_prep_ms",
                "t_cloud_prompt_ms",
                "t_edge_distill_ms",
                "t_final_ms",
                "total_compute_ms",
            ]
            + bw_cols
        )
        for task, (m, _) in summary_by_task.items():
            total_kb = m["upload_kb"] + m["download_kb"]
            bw_vals = [m["total_compute_ms"] + comm_ms(total_kb, bw) for bw in bandwidths]
            writer.writerow(
                [
                    task,
                    round(m["upload_kb"], 4),
                    round(m["download_kb"], 4),
                    round(m["t_edge_prep_ms"], 4),
                    round(m["t_cloud_prompt_ms"], 4),
                    round(m["t_edge_distill_ms"], 4),
                    round(m["t_final_ms"], 4),
                    round(m["total_compute_ms"], 4),
                ]
                + [round(v, 4) for v in bw_vals]
            )


def parse_args():
    p = argparse.ArgumentParser("Generate retrieval E2E summary tables")
    p.add_argument("--i2t-csv", required=True)
    p.add_argument("--t2i-csv", required=True)
    p.add_argument("--out-csv", default="results/e2e_bench_ret/summary.csv")
    p.add_argument("--bandwidths", type=int, nargs="+", default=[1000, 100, 50, 10, 5, 1])
    return p.parse_args()


def main():
    args = parse_args()
    i2t = to_float_rows(load_csv(args.i2t_csv))
    t2i = to_float_rows(load_csv(args.t2i_csv))
    summary_by_task = {
        "i2t": summarize(i2t),
        "t2i": summarize(t2i),
    }
    print_table(summary_by_task, args.bandwidths)
    save_summary_csv(args.out_csv, summary_by_task, args.bandwidths)
    print(f"Summary CSV saved: {args.out_csv}")


if __name__ == "__main__":
    main()
