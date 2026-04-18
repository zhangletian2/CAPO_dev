#!/usr/bin/env python3
"""
Pack CAPO datasets in parallel.
- Image directories → .tar.gz (pigz accelerated)
- Small files (json, txt, mat, etc.) → copy as-is
- Processed images (new/, cloud) packed separately from originals
- sun397/SUN397 filtered to val+test only via file list
"""

import json
import os
import shutil
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

CLS = '/data4/workspace/zlt/CAPO_cls/dataset'
RET = '/data4/workspace/zlt/CAPO_ret/dataset'
SEG = '/data4/workspace/zlt/MaskCLIP/data'
OUTPUT = '/data4/workspace/zlt/capo_packed'

PIGZ_THREADS = 4
MAX_PARALLEL = 8


@dataclass
class TarTask:
    """A single tar.gz job."""
    label: str
    output_path: str
    base_dir: str
    targets: list = field(default_factory=list)
    file_list: str | None = None  # path to file list for -T


@dataclass
class CopyTask:
    src: str
    dst: str


@dataclass
class DatasetPack:
    name: str
    source: str
    tar_tasks: list = field(default_factory=list)
    copy_tasks: list = field(default_factory=list)


def build_cls_datasets():
    packs = []

    simple_datasets = [
        ('caltech-101', ['101_ObjectCategories'], ['new'],
         ['Annotations.tar', 'show_annotation.m', 'split_zhou_Caltech101.json']),

        ('dtd', ['images'], ['new'],
         ['imdb', 'labels', 'split_zhou_DescribableTextures.json']),

        ('eurosat', ['2750'], ['new'],
         ['split_zhou_EuroSAT.json']),

        ('fgvc_aircraft', ['images'], ['new'],
         ['examples', 'samples', 'families.txt', 'images_box.txt',
          'images_family_test.txt', 'images_family_train.txt',
          'images_family_trainval.txt', 'images_family_val.txt',
          'images_manufacturer_test.txt', 'images_manufacturer_train.txt',
          'images_manufacturer_trainval.txt', 'images_manufacturer_val.txt',
          'images_test.txt', 'images_train.txt', 'images_val.txt',
          'images_variant_test.txt', 'images_variant_train.txt',
          'images_variant_trainval.txt', 'images_variant_val.txt',
          'manufacturers.txt', 'read_class.py', 'variants.txt']),

        ('food-101', ['images'], ['new'],
         ['meta', 'license_agreement.txt', 'README.txt', 'split_zhou_Food101.json']),

        ('imagenet-adversarial', ['imagenet-a'], ['new'],
         ['classnames.txt']),

        ('imagenet-rendition', ['imagenet-r'], ['new'],
         ['classnames.txt']),

        ('imagenet-sketch', ['images'], ['new'],
         ['classnames.txt']),

        ('imagenetv2', ['imagenetv2-matched-frequency-format-val'], ['new'],
         ['classnames.txt']),

        ('oxford_flowers', ['jpg'], ['new'],
         ['cat_to_name.json', 'imagelabels.mat', 'split_zhou_OxfordFlowers.json']),

        ('oxford_pets', ['images'], ['new'],
         ['annotations', 'split_zhou_OxfordPets.json']),

        ('stanford_cars', ['cars_test', 'cars_train'], ['new'],
         ['cars_test_annos_withlabels.mat', 'devkit', 'split_zhou_StanfordCars.json']),

        ('ucf101', ['UCF-101-midframes'], ['new'],
         ['split_zhou_UCF101.json']),
    ]

    for name, orig_dirs, new_dirs, copy_items in simple_datasets:
        src = f'{CLS}/{name}'
        out = f'{OUTPUT}/{name}'
        p = DatasetPack(name=name, source=src)

        for d in orig_dirs:
            p.tar_tasks.append(TarTask(
                label=f'{name}/{d}',
                output_path=f'{out}/{d}.tar.gz',
                base_dir=src,
                targets=[d],
            ))
        for d in new_dirs:
            p.tar_tasks.append(TarTask(
                label=f'{name}/{d}',
                output_path=f'{out}/{d}.tar.gz',
                base_dir=src,
                targets=[d],
            ))
        for item in copy_items:
            p.copy_tasks.append(CopyTask(
                src=f'{src}/{item}', dst=f'{out}/{item}'
            ))
        packs.append(p)

    # --- imagenet (nested: images/val, images/new) ---
    src = f'{CLS}/imagenet'
    out = f'{OUTPUT}/imagenet'
    p = DatasetPack(name='imagenet', source=src)
    p.tar_tasks.append(TarTask(
        label='imagenet/images/val',
        output_path=f'{out}/images/val.tar.gz',
        base_dir=f'{src}/images',
        targets=['val'],
    ))
    p.tar_tasks.append(TarTask(
        label='imagenet/images/new',
        output_path=f'{out}/images/new.tar.gz',
        base_dir=f'{src}/images',
        targets=['new'],
    ))
    for item in ['ILSVRC2012_devkit_t12.tar.gz', 'meta.bin']:
        p.copy_tasks.append(CopyTask(
            src=f'{src}/images/{item}', dst=f'{out}/images/{item}'
        ))
    for item in ['classnames.txt', 'ILSVRC2012_devkit_t12', 'split.py']:
        p.copy_tasks.append(CopyTask(
            src=f'{src}/{item}', dst=f'{out}/{item}'
        ))
    packs.append(p)

    # --- sun397 (SUN397/ filtered to val+test, new/ already cleaned) ---
    src = f'{CLS}/sun397'
    out = f'{OUTPUT}/sun397'
    p = DatasetPack(name='sun397', source=src)

    with open(f'{src}/split_zhou_SUN397.json') as f:
        splits = json.load(f)
    val_test_paths = [item[0] for item in splits['val'] + splits['test']]
    sun397_filelist = os.path.join(tempfile.gettempdir(), 'sun397_valtest.txt')
    with open(sun397_filelist, 'w') as f:
        for path in sorted(val_test_paths):
            f.write(f'SUN397/{path}\n')

    p.tar_tasks.append(TarTask(
        label='sun397/SUN397 (val+test)',
        output_path=f'{out}/SUN397.tar.gz',
        base_dir=src,
        file_list=sun397_filelist,
    ))
    p.tar_tasks.append(TarTask(
        label='sun397/new',
        output_path=f'{out}/new.tar.gz',
        base_dir=src,
        targets=['new'],
    ))
    copy_items = [
        'ClassName.txt', 'split10.mat', 'split.py', 'split_zhou_SUN397.json',
    ]
    for i in range(1, 11):
        copy_items.append(f'Testing_{i:02d}.txt')
        copy_items.append(f'Training_{i:02d}.txt')
    for item in copy_items:
        p.copy_tasks.append(CopyTask(
            src=f'{src}/{item}', dst=f'{out}/{item}'
        ))
    packs.append(p)

    return packs


def build_ret_datasets():
    packs = []

    # coco2014: pack image dirs, copy jsons, skip pkl
    src = f'{RET}/coco2014'
    out = f'{OUTPUT}/coco2014'
    p = DatasetPack(name='coco2014', source=src)
    for d in ['val2014', 'test2014']:
        p.tar_tasks.append(TarTask(
            label=f'coco2014/{d}',
            output_path=f'{out}/{d}.tar.gz',
            base_dir=src,
            targets=[d],
        ))
    for fn in os.listdir(src):
        full = os.path.join(src, fn)
        if os.path.isfile(full) and not fn.endswith('.pkl'):
            if fn in ('md5sum.txt', 'stat.py') or fn.endswith('.json'):
                p.copy_tasks.append(CopyTask(src=full, dst=f'{out}/{fn}'))
    packs.append(p)

    # coco2014_cloud
    src = f'{RET}/coco2014_cloud'
    out = f'{OUTPUT}/coco2014_cloud'
    p = DatasetPack(name='coco2014_cloud', source=src)
    p.tar_tasks.append(TarTask(
        label='coco2014_cloud/val2014',
        output_path=f'{out}/val2014.tar.gz',
        base_dir=src,
        targets=['val2014'],
    ))
    packs.append(p)

    return packs


def build_seg_datasets():
    src = f'{SEG}/VOCdevkit'
    out = f'{OUTPUT}/VOCdevkit'
    p = DatasetPack(name='VOCdevkit', source=src)
    p.tar_tasks.append(TarTask(
        label='VOCdevkit/VOC2012',
        output_path=f'{out}/VOC2012.tar.gz',
        base_dir=src,
        targets=['VOC2012'],
    ))
    return [p]


def run_tar(task: TarTask) -> str:
    os.makedirs(os.path.dirname(task.output_path), exist_ok=True)
    compress = f'pigz -p {PIGZ_THREADS}'
    if task.file_list:
        cmd = ['tar', '-I', compress, '-cf', task.output_path,
               '-C', task.base_dir, '-T', task.file_list]
    else:
        cmd = ['tar', '-I', compress, '-cf', task.output_path,
               '-C', task.base_dir] + task.targets

    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - t0

    size = os.path.getsize(task.output_path) if os.path.exists(task.output_path) else 0
    size_str = f'{size / (1024**3):.2f}G' if size > 1024**3 else f'{size / (1024**2):.1f}M'

    if result.returncode != 0:
        return f'  FAIL  {task.label} ({elapsed:.1f}s) — {result.stderr.strip()}'
    return f'  OK    {task.label} → {size_str} ({elapsed:.1f}s)'


def run_copies(pack: DatasetPack):
    out_dir = f'{OUTPUT}/{pack.name}'
    os.makedirs(out_dir, exist_ok=True)
    for c in pack.copy_tasks:
        os.makedirs(os.path.dirname(c.dst), exist_ok=True)
        if os.path.isdir(c.src):
            if os.path.exists(c.dst):
                shutil.rmtree(c.dst)
            shutil.copytree(c.src, c.dst)
        elif os.path.isfile(c.src):
            shutil.copy2(c.src, c.dst)


def main():
    all_packs = build_cls_datasets() + build_ret_datasets() + build_seg_datasets()

    print(f'Output: {OUTPUT}')
    print(f'Datasets: {len(all_packs)}')

    all_tar_tasks = []
    for pack in all_packs:
        run_copies(pack)
        all_tar_tasks.extend(pack.tar_tasks)

    print(f'Tar tasks: {len(all_tar_tasks)}')
    print(f'Parallel workers: {MAX_PARALLEL}, pigz threads/worker: {PIGZ_THREADS}')
    print()

    t_start = time.time()
    with ThreadPoolExecutor(max_workers=MAX_PARALLEL) as pool:
        futures = {pool.submit(run_tar, t): t for t in all_tar_tasks}
        for future in as_completed(futures):
            print(future.result())

    elapsed = time.time() - t_start
    total_size = sum(
        os.path.getsize(t.output_path)
        for t in all_tar_tasks if os.path.exists(t.output_path)
    )
    print(f'\nTotal tar.gz: {total_size / (1024**3):.2f} GiB')
    print(f'Elapsed: {elapsed:.0f}s')


if __name__ == '__main__':
    main()
