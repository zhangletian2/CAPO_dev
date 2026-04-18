import os
import numpy as np
import zlib
import lzma
import bz2
import gzip
import pickle
import pandas as pd
from tqdm import tqdm
from PIL import Image

feat_dir = '/share/project/xiaotongli/tta/data/imagenetv2/imagenetv2-matched-frequency-format-val'
bin_dir = '/share/project/xiaotongli/tta/data/enc2dec/intermediate/imagenet_v'
rec_dir = '/share/project/xiaotongli/tta/data/imagenetv2/new'

if __name__ == '__main__':
    all_imgs = []
    for subdir, _, files in os.walk(feat_dir):
        # 只处理直接在 root_folder 下的文件和子文件夹中的文件
        if subdir == feat_dir:
            # 根目录下的文件直接处理
            files_to_process = files
            relative_subdir = ''
        else:
            # 子目录中的文件处理
            files_to_process = files
            relative_subdir = os.path.relpath(subdir, feat_dir)

        for file in files_to_process:
            # 检查文件是否是图片（jpg或png格式）
            if file.lower().endswith(('jpg', 'png', 'jpeg', 'JPEG')):
                all_imgs.append((subdir, file, relative_subdir))
    comp_size = []
    comp_r = []
    mse = []
    cnt = 0
    with tqdm(total=len(all_imgs), desc="Calculating", unit="file") as pbar:
        for subdir, file, relative_subdir in all_imgs:
            ori_path = os.path.join(subdir, file)
            bin_path = os.path.join(bin_dir, relative_subdir, file.split('.')[0] + '.bin')
            rec_path = os.path.join(rec_dir, relative_subdir, file)
            ori_data = np.array(Image.open(ori_path))
            rec_data = np.array(Image.open(rec_path))
            original_size = ori_data.nbytes

            compressed_size = os.path.getsize(bin_path)
            comp_size.append(compressed_size)
            compression_ratio = compressed_size / original_size
            err = np.mean((ori_data - rec_data) ** 2)
            comp_r.append(compression_ratio)
            mse.append(err)
            pbar.update(1)

    print(f'Avg Bin Size: {np.average(comp_size)}, Avg Compression Ratio: {np.average(comp_r)}, Avg MSE: {np.average(mse)}')
