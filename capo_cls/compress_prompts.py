"""Compress saved student prompts with zlib and report size before/after."""

import os
import zlib
import argparse
import numpy as np


def compress_one(npy_path):
    raw = open(npy_path, "rb").read()
    compressed = zlib.compress(raw)
    out_path = npy_path + ".zlib"
    with open(out_path, "wb") as f:
        f.write(compressed)
    return len(raw), len(compressed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt_dir", type=str, default="./saved_prompts",
                        help="root directory of saved prompts")
    args = parser.parse_args()

    total_raw, total_comp = 0, 0
    rows = []

    for root, _, files in sorted(os.walk(args.prompt_dir)):
        for fname in sorted(files):
            if not fname.endswith(".npy"):
                continue
            path = os.path.join(root, fname)
            raw_size, comp_size = compress_one(path)
            ratio = comp_size / raw_size * 100
            rel = os.path.relpath(path, args.prompt_dir)
            rows.append((rel, raw_size, comp_size, ratio))
            total_raw += raw_size
            total_comp += comp_size

    if not rows:
        print(f"No .npy files found in {args.prompt_dir}")
        return

    name_w = max(len(r[0]) for r in rows)
    header = f"{'File':<{name_w}}  {'Raw (B)':>10}  {'Zlib (B)':>10}  {'Ratio':>7}"
    print(header)
    print("-" * len(header))
    for rel, raw, comp, ratio in rows:
        print(f"{rel:<{name_w}}  {raw:>10,}  {comp:>10,}  {ratio:>6.1f}%")

    print("-" * len(header))
    total_ratio = total_comp / total_raw * 100 if total_raw else 0
    print(f"{'TOTAL':<{name_w}}  {total_raw:>10,}  {total_comp:>10,}  {total_ratio:>6.1f}%")
    print(f"\nCompressed files saved as *.npy.zlib alongside originals.")


if __name__ == "__main__":
    main()
