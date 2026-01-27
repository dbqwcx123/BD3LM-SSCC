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
from PIL import Image
import matplotlib.pyplot as plt

from torch.utils.data import IterableDataset, get_worker_info
import torch.distributed as dist
from natsort import natsorted
import random
import imageio
from torchvision import transforms

import torch
import typing
import math

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


class Div2kPatchDataset(IterableDataset):
    def __init__(self, data_path, tokenizer, samples_per_image=50, is_channel_wised=False, shuffle=True, split='train'):
      super().__init__()
      self.data_path = data_path
      self.tokenizer = tokenizer
      self.samples_per_image = samples_per_image
      self.is_channel_wised = is_channel_wised
      self.shuffle = shuffle
      self.split = split
      
      # --- 训练集启用数据增广 ---
      if split == 'train':
        self.transform = transforms.Compose([
          # 改为 RandomCrop，保留原始分辨率特征，也符合 Block 处理逻辑
          transforms.RandomCrop(size=(32, 32)),
          transforms.RandomHorizontalFlip(p=0.5),
          transforms.RandomVerticalFlip(p=0.5),
        ])
      else:
        self.transform = transforms.Compose([
          # transforms.Resize((32, 32)),
          transforms.CenterCrop(size=(32, 32)),
    ])
      # ------------------------
      
      # 1. 预先加载所有文件路径 (移出 __iter__ 以便切分)
      if not os.path.exists(data_path):
        raise ValueError(f"Data path {data_path} does not exist.")
      
      self.all_files = [
        os.path.join(data_path, item) 
        for item in os.listdir(data_path) 
        if item.lower().endswith('.png')
      ]
      self.all_files = natsorted(self.all_files)
      if shuffle:
        random.seed(42) 
        random.shuffle(self.all_files)
      print(f"Dataset initialized: Found {len(self.all_files)} images.")

    def process_patch_to_tokens(self, patch):
      flat_pixels = patch.flatten() 
      num_str_tokens = [str(val) for val in flat_pixels]
      input_ids = self.tokenizer.convert_tokens_to_ids(num_str_tokens)
      return torch.tensor(input_ids, dtype=torch.long)

    def _get_image_iterator(self, files):
      """内部生成器：处理指定的文件列表"""
      for file_path in files:
        image_np = imageio.v2.imread(file_path)
        # --- 决定这张图产出多少个样本 ---
        # 训练集：由 samples_per_image 决定
        # 验证集：每张图只产出 1 个样本（就是它自己）
        num_samples = self.samples_per_image if self.split == 'train' else 1
        
        for _ in range(num_samples):
          image_pil = Image.fromarray(image_np)
          image_pil = self.transform(image_pil)
          patch = np.array(image_pil)
          
          # 按通道拆分
          patches_to_yield = []
          if self.is_channel_wised:
            for i in range(patch.shape[-1]):
              patches_to_yield.append(patch[:, :, i:i+1]) # (32, 32, 1)
          else:
            patches_to_yield.append(patch)

          # Yield 数据
          for p in patches_to_yield:
            input_ids = self.process_patch_to_tokens(p)
            attention_mask = torch.ones_like(input_ids, dtype=torch.long)
            yield {
              "input_ids": input_ids,
              "attention_mask": attention_mask
            }

    def __iter__(self):
      # --- 处理多卡 DDP 环境下的数据切分 ---
      if dist.is_initialized():
        num_gpus = dist.get_world_size()
        gpu_id = dist.get_rank()
      else:
        num_gpus = 1
        gpu_id = 0

      # 把总文件列表按 GPU 数量均分
      per_gpu_files_count = int(math.ceil(len(self.all_files) / float(num_gpus)))
      start_gpu = gpu_id * per_gpu_files_count
      end_gpu = min(start_gpu + per_gpu_files_count, len(self.all_files))
      files_on_this_gpu = self.all_files[start_gpu:end_gpu]
      
      # --- 处理单卡内的多 Worker 切分 ---
      worker_info = get_worker_info()
      
      if worker_info is None:
        files_to_process = files_on_this_gpu
      else:
        # 多进程模式：基于当前 GPU 分到的文件，再分给每个 worker
        per_worker = int(math.ceil(len(files_on_this_gpu) / float(worker_info.num_workers)))
        worker_id = worker_info.id
        start = worker_id * per_worker
        end = min(start + per_worker, len(files_on_this_gpu))
        files_to_process = files_on_this_gpu[start:end]
      
      return self._get_image_iterator(files_to_process)

    def __len__(self):
      """
      计算总样本数，用于 tqdm 进度条显示。
      """
      num_files = len(self.all_files)
      
      if self.split == 'train':
        multiplier = self.samples_per_image
      else:
        multiplier = 1 # 验证集每张图只测一次
          
      # 如果是按通道拆分，数量还要 * 3
      if self.is_channel_wised:
        multiplier *= 3
          
      return num_files * multiplier


class RandomFaultTolerantSampler(torch.utils.data.RandomSampler):

  def __init__(self, *args, generator=None, **kwargs):
    # TD [2022-07-17]: We don't force the seed to be zero. We generate random seed,
    # which should be reproducible if pl.seed_everything was called beforehand.
    # This means that changing the seed of the experiment will also change the
    # sampling order.
    if generator is None:
      seed = int(torch.empty((), dtype=torch.int64).random_().item())
      generator = torch.Generator().manual_seed(seed)
    kwargs.pop('shuffle', None)
    super().__init__(*args, generator=generator, **kwargs)
    self.counter = 0
    self.restarting = False

  def state_dict(self):
    return {'random_state': self.generator.get_state(),
            'counter': self.counter}

  def load_state_dict(self, state_dict):
    self.generator.set_state(state_dict.get('random_state'))
    self.counter = state_dict['counter']
    # self.start_counter = self.counter
    self.restarting = True

  # TD [2022-08-28] Setting the len will cause PL to think there are only a few batches left per
  # epoch, and subsequent epoch will have very few batches.

  def __iter__(self) -> typing.Iterator[int]:
    n = len(self.data_source)

    self.state = self.generator.get_state()
    indices = torch.randperm(n, generator=self.generator).tolist()

    if not self.restarting:
      self.counter = 0
    else:
      indices = indices[self.counter:]
      self.restarting = False

    for index in indices:
      self.counter += 1
      yield index

    self.counter = 0

class FaultTolerantDistributedSampler(torch.utils.data.DistributedSampler):

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.counter = 0
    self.restarting = False

  def state_dict(self):
    return {'epoch': self.epoch, 'counter': self.counter}

  def load_state_dict(self, state_dict):
    self.epoch = state_dict['epoch']
    self.counter = state_dict['counter']
    self.restarting = True

  # TD [2022-08-28] Setting the len will cause PL to think there are only a few batches left per
  # epoch, and subsequent epoch will have very few batches.
  def __iter__(self):
    if self.shuffle:
      # deterministically shuffle based on epoch and seed
      g = torch.Generator()
      g.manual_seed(self.seed + self.epoch)
      indices = torch.randperm(len(self.dataset), generator=g).tolist()  # type: ignore[arg-type]
    else:
      indices = list(range(len(self.dataset)))  # type: ignore[arg-type]

    if not self.drop_last:
      # add extra samples to make it evenly divisible
      padding_size = self.total_size - len(indices)
      if padding_size <= len(indices):
        indices += indices[:padding_size]
      else:
        indices += (indices * math.ceil(
          padding_size / len(indices)))[:padding_size]
    else:
      # remove tail of data to make it evenly divisible.
      indices = indices[:self.total_size]
    assert len(indices) == self.total_size

    # subsample
    indices = indices[self.rank:self.total_size:self.num_replicas]
    assert len(indices) == self.num_samples

    if not self.restarting:
      self.counter = 0
    else:
      indices = indices[self.counter:]
      self.restarting = False

    for index in indices:
      self.counter += 1
      yield index

    self.counter = 0
