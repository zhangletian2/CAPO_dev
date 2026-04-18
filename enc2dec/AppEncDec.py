# The implementation of GDN is inherited from
# https://github.com/jorge-pessoa/pytorch-gdn,
# under the MIT License. The source code is 
# also related to an implementation
# of the arithmetic coding by Nayuki from
# https://github.com/nayuki/Reference-arithmetic-coding
# under the MIT License.
#
# This file is being made available under the BSD License.  
# Copyright (c) 2021 Yueyu Hu

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import glob

import numpy as np
import pickle

import math
import time

from networks import *

if __name__ == "__main__":
  parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)

  parser.add_argument(
      "--command", choices=["compress", "decompress", 'all'],
      help="What to do? Choose from `compress` and `decompress`")
  parser.add_argument(
      "--input", nargs="?", default=None,
      help="Input filename.")
  parser.add_argument(
      "--intermediate", nargs="?",default=None,
      help="Intermediate filename.")
  parser.add_argument(
      "--output", nargs="?",default=None,
      help="Output filename.")
  parser.add_argument(
      "--json", nargs="?",default=None,
      help="Data split json filename if need.")
  parser.add_argument(
      "--json_key", default=None, type=str,
      help="Key in split_zhou JSON to extract, e.g. 'test', 'val', 'train'.")
  parser.add_argument(
      "--txt_filter", default=None, type=str,
      help="Text file with image IDs (one per line), e.g. aircraft images_variant_test.txt.")
  parser.add_argument(
      "--qp", default=1, type=int,
      help="Quality parameter, choose from [1~7] (model0) or [1~8] (model1)"
  )
  parser.add_argument(
      "--model_type", default=0, type=int,
      help="Model type, choose from 0:PSNR 1:MS-SSIM"
  )
  parser.add_argument(
      "--save_recon", default=0, type=int,
      help="Whether to save reconstructed image in the encoding process."
  )

  parser.add_argument(
      "--device", default='cpu', type=str,
      help="Which device does the network run on?"
  )
  parser.add_argument(
      "--shard", default=None, type=str,
      help="Process only shard i of N total shards, format: 'i/N' (0-indexed). "
           "E.g. --shard 0/4 processes the first quarter of images."
  )

  args = parser.parse_args()

  if args.command == "compress":
    if args.input is None or args.output is None:
      raise ValueError("Need input and output filename for compression.")
    compress_low(args)
  elif args.command == "decompress":
    if args.input is None or args.output is None:
      raise ValueError("Need input and output filename for decompression.")
    decompress_low(args)
  elif args.command == "all":
    if args.input is None or args.output is None:
      raise ValueError("Need input and output filename for decompression.")
    compress_decompress(args)
