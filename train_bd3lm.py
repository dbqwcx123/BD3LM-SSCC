import os
import fsspec
import hydra
import lightning as L
import omegaconf
import rich.syntax
import rich.tree
import torch
import transformers
from torch.utils.data import IterableDataset, get_worker_info
import torch.distributed as dist
import math
from natsort import natsorted
import random
import imageio
from torchvision import transforms
from PIL import Image
import numpy as np

import diffusion_train
import utils

omegaconf.OmegaConf.register_new_resolver(
  'cwd', os.getcwd)
omegaconf.OmegaConf.register_new_resolver(
  'device_count', torch.cuda.device_count)
omegaconf.OmegaConf.register_new_resolver(
  'eval', eval)
omegaconf.OmegaConf.register_new_resolver(
  'div_up', lambda x, y: (x + y - 1) // y)


@L.pytorch.utilities.rank_zero_only
def _print_config(
  config: omegaconf.DictConfig,
  resolve: bool = True,
  save_cfg: bool = True) -> None:
  """Prints content of DictConfig using Rich library and its tree structure.
  
  Args:
    config (DictConfig): Configuration composed by Hydra.
    resolve (bool): Whether to resolve reference fields of DictConfig.
    save_cfg (bool): Whether to save the configuration tree to a file.
  """

  style = 'dim'
  tree = rich.tree.Tree('CONFIG', style=style, guide_style=style)

  fields = config.keys()
  for field in fields:
    branch = tree.add(field, style=style, guide_style=style)

    config_section = config.get(field)
    branch_content = str(config_section)
    if isinstance(config_section, omegaconf.DictConfig):
      branch_content = omegaconf.OmegaConf.to_yaml(
        config_section, resolve=resolve)

    branch.add(rich.syntax.Syntax(branch_content, 'yaml'))
  rich.print(tree)
  if save_cfg:
    with fsspec.open(
      '{}/config_tree.txt'.format(
        config.checkpointing.save_dir), 'w') as fp:
      rich.print(tree, file=fp)


@L.pytorch.utilities.rank_zero_only
def _print_batch(train_ds, valid_ds, tokenizer, k=8):
  for dl_type, dl in [
    ('train', train_ds), ('valid', valid_ds)]:
    print(f'Printing {dl_type} dataloader batch.')
    batch = next(iter(dl))
    print('Batch input_ids.shape', batch['input_ids'].shape)
    first = batch['input_ids'][0, :k]
    last = batch['input_ids'][0, -k:]
    print(f'First {k} tokens:', tokenizer.decode(first))
    print('ids:', first)
    print(f'Last {k} tokens:', tokenizer.decode(last))
    print('ids:', last)


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
          transforms.RandomCrop(size=(32, 32)),
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

def _train(config, logger, tokenizer):
  logger.info('Starting Training.')
  wandb_logger = None
  if config.get('wandb', None) is not None:
    wandb_logger = L.pytorch.loggers.WandbLogger(
      config=omegaconf.OmegaConf.to_object(config),
      ** config.wandb)

  if (config.checkpointing.resume_from_ckpt
      and config.checkpointing.resume_ckpt_path is not None
      and utils.fsspec_exists(
        config.checkpointing.resume_ckpt_path)):
    ckpt_path = config.checkpointing.resume_ckpt_path
    logger.info(f'Resuming training at {ckpt_path}')
  else:
    ckpt_path = None

  # Lightning callbacks
  callbacks = []
  if 'callbacks' in config:
    for _, callback in config.callbacks.items():
      callbacks.append(hydra.utils.instantiate(callback))

  # import dataloader
  # train_ds, valid_ds = dataloader.get_dataloaders(config, tokenizer)
  
  train_set = Div2kPatchDataset(
    data_path=os.path.join(config.loader.data_dir, "train_small"),
    tokenizer=tokenizer,
    samples_per_image=1,
    is_channel_wised=config.data.is_channel_wised,
    shuffle=True,
    split='train'
  )

  valid_set = Div2kPatchDataset(
    data_path=os.path.join(config.loader.data_dir, "test"),
    tokenizer=tokenizer,
    samples_per_image=1,
    is_channel_wised=config.data.is_channel_wised,
    shuffle=False,
    split='test'
  )
  
  train_loader = torch.utils.data.DataLoader(
    train_set,
    batch_size=config.loader.batch_size,
    num_workers=config.loader.num_workers,
    pin_memory=config.loader.pin_memory,
    shuffle=False,  # 必须为 False，乱序已在 Dataset 内部通过 shuffle=True 处理
    persistent_workers=True if config.loader.num_workers > 0 else False
  )

  valid_loader = torch.utils.data.DataLoader(
    valid_set,
    batch_size=config.loader.eval_batch_size,
    num_workers=config.loader.num_workers,
    pin_memory=config.loader.pin_memory,
    shuffle=False, 
    persistent_workers=True if config.loader.num_workers > 0 else False
  )

  # 补充原 dataloader 的特殊属性
  train_loader.tokenizer = tokenizer
  valid_loader.tokenizer = tokenizer
  
  # _print_batch(train_loader, valid_loader, tokenizer)

  
  logger.info(f'Loading pretrained model from {config.training.from_pretrained}')
  # load pretraining checkpoint (local)
  model = diffusion_train.Diffusion(config=config, tokenizer=tokenizer)
  # add buffers for grid search
  model.register_buffer('sampling_eps_min', torch.tensor(
    config.training.sampling_eps_min))
  model.register_buffer('sampling_eps_max', torch.tensor(
    config.training.sampling_eps_max))
  
  # torch.set_float32_matmul_precision('high')
  
  trainer = hydra.utils.instantiate(
    config.trainer,
    default_root_dir=os.getcwd(),
    callbacks=callbacks,
    strategy=hydra.utils.instantiate(config.strategy),
    logger=wandb_logger)

  trainer.fit(model, train_loader, valid_loader, ckpt_path=ckpt_path)


@hydra.main(version_base=None, config_path='configs',
            config_name='config_train')
def main(config):
  """Main entry point for training."""
  L.seed_everything(config.seed)
  # _print_config(config, resolve=True, save_cfg=True)
  
  logger = utils.get_logger(__name__)
  tokenizer = utils.get_tokenizer(config)

  _train(config, logger, tokenizer)



if __name__ == '__main__':
  main()