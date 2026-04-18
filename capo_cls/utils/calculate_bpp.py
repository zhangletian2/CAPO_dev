# 重新导入必要的库
import os
import zlib
import lzma
import bz2
import gzip
import torch
import torch.nn as nn
import torch.distributions as dist
import numpy as np
import pickle
import sys
from tqdm import tqdm

# 定义熵瓶颈模型
class GaussianModel(nn.Module):
    def __init__(self):
        super(GaussianModel, self).__init__()
        self.m_normal_dist = dist.Normal(0., 1.)

    def _cumulative(self, inputs, stds, mu):
        half = 0.5
        eps = 1e-6
        upper = (inputs - mu + half) / stds
        lower = (inputs - mu - half) / stds
        cdf_upper = self.m_normal_dist.cdf(upper)
        cdf_lower = self.m_normal_dist.cdf(lower)
        res = cdf_upper - cdf_lower
        return res

    def forward(self, inputs, stds, mu):
        likelihood = self._cumulative(inputs, stds, mu)
        likelihood_bound = 1e-8
        likelihood = torch.clamp(likelihood, min=likelihood_bound)
        return likelihood


# 定义特征压缩模块
class FeatureCompressor(nn.Module):
    def __init__(self, in_dim):
        super(FeatureCompressor, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )
        self.decoder = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, in_dim)
        )
        self.entropy_bottleneck = GaussianModel()
        self.sigma = nn.Parameter(torch.ones((1, 128)) * 0.1, requires_grad=True)

    def forward(self, x):
        # 编码
        z = self.encoder(x)
        z_rounded = torch.round(z)  # 模拟量化
        noise = torch.rand_like(z_rounded) - 0.5  # 添加噪声
        z_noisy = z + noise

        # 概率建模
        mu = torch.zeros_like(self.sigma)
        sigma = torch.clamp(self.sigma, min=1e-6)
        likelihood = self.entropy_bottleneck(z_noisy, sigma, mu)

        # 比特率计算
        num_features = x.numel()
        bpp = torch.log(likelihood).sum() / (-torch.log(torch.tensor(2.0)) * num_features)

        # 解码
        # reconstructed = self.decoder(z_rounded)
        reconstructed = None
        return reconstructed, bpp


def lossless_comp(data):
    packed_bytes = pickle.dumps(data)
    compressed_data = zlib.compress(packed_bytes)
    compressed_size = sys.getsizeof(compressed_data)
    return compressed_size


if __name__ == "__main__":
    # 将 numpy 数组转换为 PyTorch 张量
    root_folder = './'
    original_size = 0
    compressed_size = 0
    for file in tqdm(os.listdir(root_folder)):
        if file.lower().endswith('npy'):
            input_numpy = np.load(os.path.join(root_folder, file), allow_pickle=True)
            original_size += input_numpy.nbytes
            input_tensor = torch.tensor(input_numpy, dtype=torch.float32)
            compressor = FeatureCompressor(input_tensor.shape[-1])
            _, bpp = compressor(input_tensor)
            compressed_size += input_numpy.size * bpp.item()
            print("Bits Per Pixel (BPP): ", bpp.item())
            # compressed_size += lossless_comp(input_numpy)
    print("Total Bytes of parameters (4xParameters): ", original_size)
    print("Compressed Bytes of parameters: ", compressed_size)

