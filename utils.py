"""Console logger utilities.

Copied from https://github.com/HazyResearch/transformers/blob/master/src/utils/utils.py
Copied from https://docs.python.org/3/howto/logging-cookbook.html#using-a-context-manager-for-selective-logging
"""

import csv
import functools
import logging
import math

import fsspec
import lightning
import torch
from timm.scheduler import CosineLRScheduler
from torch.optim.lr_scheduler import LambdaLR


def count_parameters(model):
  return sum(p.numel()
             for p in model.parameters()
             if p.requires_grad)

def fsspec_exists(filename):
  """Check if a file exists using fsspec."""
  fs, _ = fsspec.core.url_to_fs(filename)
  return fs.exists(filename)


def fsspec_listdir(dirname):
  """Listdir in manner compatible with fsspec."""
  fs, _ = fsspec.core.url_to_fs(dirname)
  return fs.ls(dirname)


def fsspec_mkdirs(dirname, exist_ok=True):
  """Mkdirs in manner compatible with fsspec."""
  fs, _ = fsspec.core.url_to_fs(dirname)
  fs.makedirs(dirname, exist_ok=exist_ok)


def print_nans(tensor, name):
  if torch.isnan(tensor).any():
    print(name, tensor)


class CosineDecayWarmupLRScheduler(
  CosineLRScheduler,
  torch.optim.lr_scheduler._LRScheduler):
  """Wrap timm.scheduler.CosineLRScheduler
  Enables calling scheduler.step() without passing in epoch.
  Supports resuming as well.
  Adapted from:
    https://github.com/HazyResearch/hyena-dna/blob/main/src/utils/optim/schedulers.py
  """

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self._last_epoch = -1
    self.step(epoch=0)

  def step(self, epoch=None):
    if epoch is None:
      self._last_epoch += 1
    else:
      self._last_epoch = epoch
    # We call either step or step_update, depending on
    # whether we're using the scheduler every epoch or every
    # step.
    # Otherwise, lightning will always call step (i.e.,
    # meant for each epoch), and if we set scheduler
    # interval to "step", then the learning rate update will
    # be wrong.
    if self.t_in_epochs:
      super().step(epoch=self._last_epoch)
    else:
      super().step_update(num_updates=self._last_epoch)


class ThreePhaseLRScheduler(LambdaLR):
    """
    实现了 Warmup -> Constant -> Cosine Decay 的三阶段学习率调度器。
    """
    def __init__(self, optimizer, total_steps, warmup_ratio=0.1, constant_ratio=0.3, last_epoch=-1, verbose=False, **kwargs):
        # **kwargs: 接收并忽略 YAML 中可能残留的其他参数 (如 t_in_epochs, warmup_lr_init 等)
        
        self.total_steps = int(total_steps)
        self.warmup_steps = int(total_steps * warmup_ratio)
        self.constant_end_step = int(total_steps * (warmup_ratio + constant_ratio))
        
        # [核心修复] 
        # 移除 verbose 参数的传递，因为旧版 PyTorch 的 LambdaLR 不支持它。
        # 即使新版支持，LambdaLR 的 verbose 打印通常不是必须的。
        super().__init__(optimizer, self.lr_lambda, last_epoch)

    def lr_lambda(self, current_step):
        # 阶段 1: Warmup (线性增长 0 -> 1)
        if current_step < self.warmup_steps:
            return float(current_step) / float(max(1, self.warmup_steps))
        
        # 阶段 2: Constant (保持 1.0)
        elif current_step < self.constant_end_step:
            return 1.0
        
        # 阶段 3: Decay (余弦衰减 1.0 -> 0.0)
        else:
            decay_steps = self.total_steps - self.constant_end_step
            current_decay_step = current_step - self.constant_end_step
            progress = float(current_decay_step) / float(max(1, decay_steps))
            
            if progress > 1.0: 
                return 0.0
            
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

class LoggingContext:
  """Context manager for selective logging."""
  def __init__(self, logger, level=None, handler=None, close=True):
    self.logger = logger
    self.level = level
    self.handler = handler
    self.close = close

  def __enter__(self):
    if self.level is not None:
      self.old_level = self.logger.level
      self.logger.setLevel(self.level)
    if self.handler:
      self.logger.addHandler(self.handler)

  def __exit__(self, et, ev, tb):
    if self.level is not None:
      self.logger.setLevel(self.old_level)
    if self.handler:
      self.logger.removeHandler(self.handler)
    if self.handler and self.close:
      self.handler.close()


def get_logger(name=__name__, level=logging.INFO) -> logging.Logger:
  """Initializes multi-GPU-friendly python logger."""

  logger = logging.getLogger(name)
  logger.setLevel(level)

  # this ensures all logging levels get marked with the rank zero decorator
  # otherwise logs would get multiplied for each GPU process in multi-GPU setup
  for level in ('debug', 'info', 'warning', 'error',
                'exception', 'fatal', 'critical'):
    setattr(logger,
            level,
            lightning.pytorch.utilities.rank_zero_only(
              getattr(logger, level)))

  return logger

def get_tokenizer(config):
    import transformers
    tokenizer = transformers.AutoTokenizer.from_pretrained(config.data.tokenizer_name_or_path)
    if tokenizer.bos_token is None and tokenizer.cls_token is not None:
        tokenizer.bos_token = tokenizer.cls_token
    if tokenizer.eos_token is None and tokenizer.sep_token is not None:
        tokenizer.eos_token = tokenizer.sep_token
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    return tokenizer
