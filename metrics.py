import torch
from torch import Tensor
import torchmetrics
import typing
from typing import Union
import transformers
import os
import torch.nn.functional as F
from tqdm import tqdm
import math

LOG2 = torch.log(torch.tensor(2.0))

class NLL(torchmetrics.aggregation.MeanMetric):
  pass

class BPD(NLL):
  def compute(self) -> Tensor:
    """Computes the bits per dimension.

    Returns:
      bpd
    """
    return self.mean_value / self.weight / LOG2

class Perplexity(NLL):
  def compute(self) -> Tensor:
    """Computes the Perplexity.

    Returns:
      Perplexity
    """
    return torch.exp(self.mean_value / self.weight)

class NFEs(torchmetrics.aggregation.MeanMetric):
  pass

class Metrics:
  def __init__(self, config=None) -> None:
    self.config=config
    metrics = torchmetrics.MetricCollection({
        'nll': NLL(), 'bpd': BPD(), 'ppl': Perplexity()})  # bpd: bits per dimension
    if hasattr(config, 'block_size'):
      self.block_size = config.block_size
    else:
      self.block_size = config.model.length
    
    self.nfes = NFEs()  # 生成样本时模型被调用的次数
    self.train_nlls = metrics.clone(prefix='train/')
    self.valid_nlls = metrics.clone(prefix='val/')

    self.sampling_eps = config.training.sampling_eps
    if getattr(config.algo, 'clip_search_delta', None):
      self.clip_search_delta = config.algo.clip_search_delta
    self.valid_vars = {self.sampling_eps: []}
    if getattr(config.algo, 'var_min', None):
      self.valid_vars = self.init_valid_vars()  # 方差
    self.eval_ppl_batch_size = \
     self.config.eval.perplexity_batch_size
    self.gen_ppl_eval_model_name_or_path = \
      config.eval.gen_ppl_eval_model_name_or_path
    self.tokenizer = transformers.AutoTokenizer.\
      from_pretrained(self.gen_ppl_eval_model_name_or_path)
    if self.tokenizer.pad_token is None:
      self.tokenizer.pad_token = self.tokenizer.eos_token
      self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

  def init_valid_vars(self):
    eps = self.sampling_eps
    if self.block_size > 1:
      eps = self.sampling_eps
      self.valid_vars = {(eps, 1): []}
      for width in self.config.algo.clip_search_widths:
        for i in torch.arange(0, 1 - width + self.clip_search_delta, self.clip_search_delta):
          min = torch.clamp(i, min=self.sampling_eps).item()
          max = torch.clamp(i + width, min=self.sampling_eps).item()
          self.valid_vars[(min, max)] = []
    else:
      eps = self.sampling_eps
      self.valid_vars = {
        (eps, 1): [],
        (1, 1): []}

  def to(self, *args, **kwargs):
    self.train_nlls = self.train_nlls.to(*args, **kwargs)
    self.valid_nlls = self.valid_nlls.to(*args, **kwargs)
    self.nfes = self.nfes.to(*args, **kwargs)

  def reset(self):
    self.train_nlls.reset()
    self.valid_nlls.reset()
    self.nfes.reset()
    if getattr(self.config.algo, 'var_min', None):
      self.init_valid_vars()
