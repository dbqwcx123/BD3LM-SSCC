# Copyright 2023 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Implements data loaders."""
from glob import glob
import pdb

from einops import einops
from natsort import natsorted
from collections.abc import Iterator
import itertools
import os.path

import numpy as np
import imageio
from torch.utils.data import DataLoader
from transformers import ImageGPTImageProcessor, ImageGPTForCausalImageModeling
from PIL import Image
import matplotlib.pyplot as plt
import sys
import torchvision


def _extract_image_patches(image: np.ndarray, patch_size: int) -> Iterator[bytes]:
    h, w = patch_size, patch_size
    height, width = image.shape[0], image.shape[1]
    for row, col in itertools.product(range(height // h), range(width // w)):  # 效果等同于两个嵌套for循环
        yield image[row * h: (row + 1) * h, col * w: (col + 1) * w]
        

def _extract_image_sequence(image: np.ndarray, patch_size: int) -> Iterator[bytes]:
    h, w = patch_size, patch_size
    height, width = image.shape[0], image.shape[1]
    total_pixels = height * width
    sequence_length = h * w
    total_chunks = total_pixels // sequence_length
    image_sequence = image.reshape(-1, image.shape[-1])
    for i in range(total_chunks):
        temp_sequence = image_sequence[i * sequence_length: (i + 1) * sequence_length]
        yield temp_sequence.reshape(h, w, image.shape[-1])



def _get_image_dataset(data_path):
    """
    遍历数据集目录，逐张读取图像
    """
    if not os.path.exists(data_path):
        raise ValueError(f"Data path {data_path} does not exist.")
    
    img_files = [os.path.join(data_path, item) for item in os.listdir(data_path)]
    img_files = natsorted(img_files)
    
    
    filet_count = 0
    for file in img_files:
        test = imageio.imread_v2(file)

        # 检查图像是否为三通道，即RGB图像
        if test.shape[-1] != 3:
            continue

        yield test, filet_count  # 逐个生成数据，返回图像和编号（类似于逐个return）
        filet_count += 1


def get_image_iterator(
        patch_size: int = -1,
        num_chunks: int = -1,
        is_channel_wised: bool = True,
        is_seq: bool = False,
        data_path: str = None,
) -> Iterator[bytes]:
    """
    获取数据集的 Patch 迭代器
    """

    image_dataset = _get_image_dataset(data_path)
    idx = 0
    image_extractor = _extract_image_sequence if is_seq else _extract_image_patches
    
    for data, img_id in image_dataset:
        if is_channel_wised:
            # 遍历3个颜色通道 (R, G, B)
            for i in range(data.shape[-1]):
                temp_data = data[:, :, i:i+1]
                for patch in image_extractor(temp_data, patch_size):
                    if idx >= num_chunks and num_chunks > 0: # 增加 num_chunks > 0 判断，方便全量训练
                        return
                    yield patch, img_id
                    idx += 1
        else:
            # 整体 RGB 处理 (H, W, 3) -> (16, 16, 3) patches
            for patch in image_extractor(data, patch_size):
                if idx >= num_chunks and num_chunks > 0:
                    return
                yield patch, img_id
                idx += 1


def patch_visualize(patch_data, save_path, patch_name):
    """
    将图像块可视化并保存为RGB图。

    参数:
    patch_data (np.ndarray): 形状为 (h, w, 3) 的RGB图像块数据
    """
    patch_data = np.asarray(patch_data)
    
    # 正确的错误检查
    if patch_data.ndim != 3 or patch_data.shape[2] != 3:
        raise ValueError("patch_data 必须是 (h, w, 3) 的RGB图像")
    
    # 更安全的数据类型处理
    if patch_data.dtype == np.float32 or patch_data.dtype == np.float64:
        # 如果是浮点数，假设范围是0-1，转换为0-255
        if patch_data.max() <= 1.0:
            patch_data = (patch_data * 255).astype(np.uint8)
        else:
            patch_data = patch_data.astype(np.uint8)
    else:
        patch_data = patch_data.astype(np.uint8)
    
    # 使用 PIL 保存RGB图像
    img = Image.fromarray(patch_data, mode='RGB')
    img.save(f'{save_path}/patch{patch_name}.png')
    plt.close()
