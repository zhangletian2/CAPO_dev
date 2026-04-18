#!/usr/bin/env python3
"""Clean up new/ directories: keep only val+test images, delete train images."""

import json
import os
import shutil
import glob
from pathlib import Path

CLS_ROOT = '/data4/workspace/zlt/CAPO_cls/dataset'

ZHOU_DATASETS = [
    'caltech-101', 'dtd', 'eurosat', 'food-101',
    'oxford_flowers', 'oxford_pets', 'stanford_cars',
    'ucf101', 'sun397',
]


def clean_zhou_dataset(ds_name):
    ds_dir = os.path.join(CLS_ROOT, ds_name)
    new_dir = os.path.join(ds_dir, 'new')
    if not os.path.isdir(new_dir):
        print(f"  [SKIP] {ds_name}: new/ not found")
        return

    jf = glob.glob(os.path.join(ds_dir, 'split_zhou_*.json'))
    if not jf:
        print(f"  [SKIP] {ds_name}: no split json found")
        return

    with open(jf[0]) as f:
        splits = json.load(f)
    val_test_paths = set(item[0] for item in splits.get('val', []) + splits.get('test', []))

    deleted = 0
    kept = 0
    for root, dirs, files in os.walk(new_dir):
        for fn in files:
            full = os.path.join(root, fn)
            rel = os.path.relpath(full, new_dir)
            if rel in val_test_paths:
                kept += 1
            else:
                os.remove(full)
                deleted += 1

    for root, dirs, _ in os.walk(new_dir, topdown=False):
        for d in dirs:
            dp = os.path.join(root, d)
            if not os.listdir(dp):
                os.rmdir(dp)

    print(f"  {ds_name}/new: kept={kept}, deleted={deleted}")


def clean_fgvc_aircraft():
    ds_dir = os.path.join(CLS_ROOT, 'fgvc_aircraft')
    new_dir = os.path.join(ds_dir, 'new')

    val_ids = set()
    for split_file in ['images_val.txt', 'images_test.txt']:
        with open(os.path.join(ds_dir, split_file)) as f:
            for line in f:
                val_ids.add(line.strip().split()[0])

    deleted = 0
    kept = 0
    for fn in os.listdir(new_dir):
        full = os.path.join(new_dir, fn)
        if not os.path.isfile(full):
            continue
        fid = os.path.splitext(fn)[0]
        if fid in val_ids:
            kept += 1
        else:
            os.remove(full)
            deleted += 1

    print(f"  fgvc_aircraft/new: kept={kept}, deleted={deleted}")


def delete_sun397_sub():
    sub_dir = os.path.join(CLS_ROOT, 'sun397', 'sub')
    if os.path.isdir(sub_dir):
        shutil.rmtree(sub_dir)
        print(f"  sun397/sub: deleted")
    else:
        print(f"  sun397/sub: not found (already deleted?)")


if __name__ == '__main__':
    print("=== Cleaning new/ directories (keep val+test only) ===\n")

    for ds in ZHOU_DATASETS:
        clean_zhou_dataset(ds)

    clean_fgvc_aircraft()

    print()
    print("=== Deleting sun397/sub ===")
    delete_sun397_sub()

    print("\nDone.")
