import os
# os.environ['TRANSFORMERS_OFFLINE'] = '1'
# os.environ["TOKENIZERS_PARALLELISM"] = "false"
import itertools
from dataclasses import dataclass

import hydra.utils
import lightning as L
import numpy as np
import torch
import torch.nn.functional as F
import transformers
from tqdm import tqdm
from collections import OrderedDict
import numpy as np
import itertools
import math

import metrics
import models
import noise_schedule
import utils

@dataclass
class Loss:
  loss: torch.FloatTensor            # 用于反向传播的 Loss (Weighted CE)
  nlls: torch.FloatTensor            # 用于 Metric 统计的 Weighted NLL Tensor
  token_mask: torch.FloatTensor      # Mask
  unweighted_loss: torch.FloatTensor # 用于记录的 Unweighted CE Scalar

# class LightningModule(
#     _DeviceDtypeModuleMixin,
#     HyperparametersMixin,
#     ModelHooks,
#     DataHooks,
#     CheckpointHooks,
#     Module,)
class Diffusion(L.LightningModule):
  def __init__(
    self,
    config,
    tokenizer):
    super().__init__()
    self.save_hyperparameters()
    self.config = config
    self.tokenizer = tokenizer
    self.mask_index = self.tokenizer.vocab_size
    self.vocab_size = self.tokenizer.vocab_size + 1
    self.block_size = self.config.block_size
    if self.config.algo.backbone == 'dit':
      self.backbone = models.dit.DIT(
        self.config, vocab_size=self.vocab_size)
    elif self.config.algo.backbone == 'hf_dit':
      self.backbone = transformers.AutoModelForMaskedLM.from_pretrained(
        config.training.from_pretrained, trust_remote_code=True, local_files_only=True)
      # Regenerate mask if pretrained model uses flex attention mask
      # and current model uses sdpa mask
      if getattr(self.backbone.config, 'attn_backend', None) == 'flex' and \
        self.config.model.attn_backend == 'sdpa':
        self.backbone.config.attn_backend = 'sdpa'
        for i in self.backbone.backbone.blocks:
          i.attn_backend = 'sdpa'
        self.backbone.backbone.gen_mask(self.config.model.length, self.block_size, attn_backend='sdpa')
    else:
      raise ValueError(f'Unknown backbone: {self.config.algo.backbone}')

    self.T = self.config.algo.T
    self.num_tokens = self.config.model.length

    self.noise = noise_schedule.get_noise(self.config)
    self.metrics = metrics.Metrics(config)

    self.ema = models.ema.ExponentialMovingAverage(
      self._get_parameters(),
      decay=self.config.training.ema)
    
    self.register_buffer('sampling_eps_min', torch.tensor(
      self.config.training.sampling_eps_min))
    self.register_buffer('sampling_eps_max', torch.tensor(
      self.config.training.sampling_eps_max))
    
    self.neg_infinity = -1000000.0
    
    # 1. 预计算 Pixel Token IDs 和 Vocab Map
    pixel_ids_list = []
    for i in range(256):
        # 将 '0'-'255' 字符串转为对应的 ID
        pixel_ids_list.append(self.tokenizer.convert_tokens_to_ids(str(i)))
    
    # 注册 buffer，会自动随模型移动到 GPU
    self.register_buffer('pixel_token_ids', torch.tensor(pixel_ids_list, dtype=torch.long))
    
    # 创建全词表到 0-255 的映射表 (默认为 -1)
    vocab_map_tensor = torch.full((self.vocab_size,), -1, dtype=torch.long)
    for i, pid in enumerate(pixel_ids_list):
        if pid < self.vocab_size:
            vocab_map_tensor[pid] = i
    self.register_buffer('vocab_map', vocab_map_tensor)

# __init__方法中用到的自定义函数
  def _get_parameters(self):
    parameters = [self.backbone.parameters(),
                  self.noise.parameters()]
    return itertools.chain(* parameters)

# 调用父函数的基础上添加逻辑的函数
## _DeviceDtypeModuleMixin
  def to(self, *args, **kwargs):
    self = super().to(*args, **kwargs) 
    self.metrics.to(*args, **kwargs)
    if hasattr(self.backbone, "block_diff_mask") and self.config.model.attn_backend == 'sdpa':
      self.backbone.block_diff_mask = self.backbone.block_diff_mask.to(*args, **kwargs)
    elif hasattr(self.backbone, "block_diff_mask") and self.config.model.attn_backend == 'flex':
      self.backbone.block_diff_mask = self.backbone.block_diff_mask.to(self.device)
    if hasattr(self, 'sampling_eps_min') and torch.is_tensor(self.sampling_eps_min):
      self.sampling_eps_min = self.sampling_eps_min.to(*args, **kwargs)
      self.sampling_eps_max = self.sampling_eps_max.to(*args, **kwargs)
    return self

## LightningModule中定义
  def optimizer_step(self, *args, **kwargs):
    super().optimizer_step(*args, **kwargs)
    self.ema.update(self._get_parameters())


# 覆盖父函数实现自己逻辑的函数
## ModelHooks
  def on_train_start(self):
    self.ema.move_shadow_params_to_device(self.device)
      
  def on_train_epoch_start(self):
    self.backbone.train()
    self.noise.train()
    self.metrics.reset()
    assert self.metrics.train_nlls.nll.mean_value == 0
    assert self.metrics.train_nlls.nll.weight == 0

  def on_validation_epoch_start(self):
    self.metrics.reset()
    self.ema.store(itertools.chain(
      self.backbone.parameters(),
      self.noise.parameters()))
    self.ema.copy_to(itertools.chain(
      self.backbone.parameters(),
      self.noise.parameters()))
    self.eval()
    self.backbone.eval()
    self.noise.eval()
    assert self.metrics.valid_nlls.nll.mean_value == 0
    assert self.metrics.valid_nlls.nll.weight == 0
    self.sampling_eps = self.config.training.sampling_eps

  def on_validation_epoch_end(self):
    for k, v in self.metrics.valid_nlls.items():
      self.log(name=k,  value=v.compute(), on_step=False,
              on_epoch=True, sync_dist=True)
    self.ema.restore(self._get_parameters())
    if not self.trainer.sanity_checking:
      print("\nSearching clipped schedule...")
      self._clipped_schedule_search()
      
      current_min = self.sampling_eps_min.item()
      current_max = self.sampling_eps_max.item()
      self.print(f"{'='*40}")
      self.print(f"[Auto-Schedule] Epoch {self.current_epoch} Update:")
      self.print(f"  New Sampling Interval: [{current_min:.4f}, {current_max:.4f}]")
      self.print(f"{'='*40}")
      
      self.log('sampling_eps_min',
               self.sampling_eps_min,
               on_epoch=True,
               on_step=False,
               sync_dist=True,
               prog_bar=True)
      self.log('sampling_eps_max',
               self.sampling_eps_max,
               on_epoch=True,
               on_step=False,
               sync_dist=True,
               prog_bar=True)

## CheckpointHooks
  def on_load_checkpoint(self, checkpoint):
    print('Loading checkpoint at', checkpoint['global_step'])
    self.ema.load_state_dict(checkpoint['ema'])
    if 'sampling_eps_min' in checkpoint.keys():
      self.sampling_eps_min = checkpoint['sampling_eps_min']
      self.sampling_eps_max = checkpoint['sampling_eps_max']

  def on_save_checkpoint(self, checkpoint):
    checkpoint['ema'] = self.ema.state_dict()
    if hasattr(self, 'sampling_eps_min'):
      checkpoint['sampling_eps_min'] = self.sampling_eps_min
      checkpoint['sampling_eps_max'] = self.sampling_eps_max

## LightningModule中定义
  def forward(self, x, sigma, sample_mode=False, store_kv=False):
    """Returns log score."""
    sigma = self._process_sigma(sigma)
    with torch.amp.autocast('cuda', dtype=torch.float32):
      logits = self.backbone(x, sigma, sample_mode, store_kv)  # [Modified]

    x = x[:, :self.config.model.length]
    logits = logits[:, :self.config.model.length, :]
    
    return self._subs_parameterization(logits=logits, xt=x)
    # return logits

  def training_step(self, batch, batch_idx):
    del batch_idx
    losses = self._loss(batch['input_ids'],
                        batch['attention_mask'])
    self.metrics.train_nlls.update(losses.nlls, losses.token_mask)
    
    # 记录详细指标
    # 1. Weighted CE (优化目标, NLL Estimate)
    weighted_ce = losses.loss
    # 2. Unweighted CE (单纯的预测难度)
    unweighted_ce = losses.unweighted_loss
    # 3. NLL (负对数似然，单位 nats)
    nll = weighted_ce
    # 4. BPP (Bits Per Pixel)
    #    BPP = NLL (nats) / ln(2)
    bpp = weighted_ce / 0.69314718056
    
    self.log_dict({
        'train/loss': weighted_ce,
        'train/bpp': bpp
    }, on_step=True, on_epoch=False, sync_dist=True, prog_bar=True)
    
    # self.log(name='trainer/loss',
    #          value=losses.loss.item(),
    #          on_step=True,
    #          on_epoch=False,
    #          sync_dist=True)
    
    return losses.loss

  def validation_step(self, batch, batch_idx):
    for noise_clip_start in self.metrics.valid_vars.keys():
      sampling_eps_min, sampling_eps_max = noise_clip_start
      if self._check_val_sampling_intvl(sampling_eps_min, sampling_eps_max) == True:
        # compute and record nelbo
        losses_clip = self._loss(batch['input_ids'],
                          batch['attention_mask'],
                          sampling_eps_min=sampling_eps_min,
                          sampling_eps_max=sampling_eps_max)
        losses = Loss(
          nlls=losses_clip.nlls.clone(),
          token_mask=losses_clip.token_mask,
          loss=losses_clip.loss.clone(),
          unweighted_loss=losses_clip.unweighted_loss.clone())
      elif len(self.metrics.valid_vars[noise_clip_start]) < 100:
        # elbo from clipped schedule (biased estimate)
        losses_clip = self._loss(batch['input_ids'],
                          batch['attention_mask'],
                          sampling_eps_min=sampling_eps_min,
                          sampling_eps_max=sampling_eps_max)
      if len(self.metrics.valid_vars[noise_clip_start]) < 100:
        # only report variance over 100 batches
        nlls = losses_clip.nlls
        self.metrics.valid_vars[noise_clip_start].append(
          nlls.reshape(
            nlls.shape[0], -1, self.block_size).mean(-1))
    self.metrics.valid_nlls.update(losses.nlls, losses.token_mask)
    
    # 显式记录 Val 指标
    weighted_ce = losses.loss
    bpp = weighted_ce / 0.69314718056
    
    self.log_dict({
        'val/loss': weighted_ce,
        'val/bpp': bpp
    }, on_step=False, on_epoch=True, sync_dist=True)
    
    return losses.loss

  def configure_optimizers(self):
    # TODO(yair): Lightning currently giving this warning when using `fp16`:
    #  "Detected call of `lr_scheduler.step()` before `optimizer.step()`. "
    #  Not clear if this is a problem or not.
    #  See: https://github.com/Lightning-AI/pytorch-lightning/issues/5558
    optimizer = torch.optim.AdamW(
      self._get_parameters(),
      lr=self.config.optim.lr,
      betas=(self.config.optim.beta1,
             self.config.optim.beta2),
      eps=self.config.optim.eps,
      weight_decay=self.config.optim.weight_decay)

    scheduler = hydra.utils.instantiate(
      self.config.lr_scheduler, optimizer=optimizer)
    scheduler_dict = {'scheduler': scheduler,
                      'interval': 'step',
                      'monitor': 'val/loss',
                      'name': 'trainer/lr'}
    return [optimizer], [scheduler_dict]

# 自定义函数
  def _replace_ckpt_keys(self, checkpoint):
    state_dict = checkpoint['state_dict']
    new_state_dict = OrderedDict()
    for k,v in state_dict.items():
      new_state_dict[k.replace('_orig_mod.', '')] = v
    checkpoint['state_dict'] = new_state_dict
    return checkpoint

  def _subs_parameterization(self, logits, xt):
    # 修改：实现限制域归一化 (Restricted Normalization)
    # 1. 提取 0-255 对应的 logits [B, L, 256]
    logits_pixel = logits[:, :, self.pixel_token_ids]
    
    # 2. 在 256 维上进行 LogSoftmax 归一化, 确保 sum(exp(logits_pixel)) = 1
    logits_pixel = logits_pixel - torch.logsumexp(logits_pixel, dim=-1,
                                      keepdim=True)
    
    # 3. 处理未 Mask 的位置 (Conditioning)
    # 对于未被 Mask 的 Token，将其概率分布设为 One-hot (Log-prob 为 0)
    unmasked_indices = (xt != self.mask_index)
    # 将 xt (Token ID) 映射回 Pixel Index (0-255)
    xt_pixel_indices = self.vocab_map[xt]
    
    # 只有当：1. 未被 Mask  2. 且该位置确实是有效像素 Token 时，才应用强约束
    valid_unmasked = unmasked_indices & (xt_pixel_indices != -1)
    
    if valid_unmasked.any():
        # 将该位置所有类别的 log-prob 设为负无穷
        logits_pixel[valid_unmasked] = self.neg_infinity
        # 将该位置真实 Pixel 对应的 log-prob 设为 0
        logits_pixel[valid_unmasked, xt_pixel_indices[valid_unmasked]] = 0
        
    return logits_pixel # 返回维度 [B, L, 256]

  def _process_sigma(self, sigma):
    # cause of overfitting for block size 1?
    assert sigma.ndim == 2
    sigma = sigma.mean(-1).squeeze()
    if sigma.ndim == 0:
      sigma = sigma.unsqueeze(0)
    sigma = torch.zeros_like(sigma)
    assert sigma.ndim == 1, sigma.shape
    return sigma
  
  def _check_val_sampling_intvl(self, sampling_eps_min, sampling_eps_max):
    """Checks if the current sampling interval is valid for reporting likelihood."""
    if (sampling_eps_min == 1e-3 and sampling_eps_max == 1) :
      return True # elbo
    return False # not a valid elbo (biased estimate)

  def _loss(self, x0, attention_mask, t=None, sampling_eps_min=None, sampling_eps_max=None):
    if sampling_eps_min is None and hasattr(self, 'sampling_eps_min'):
      sampling_eps_min = self.sampling_eps_min
      sampling_eps_max = self.sampling_eps_max
    elif not hasattr(self, 'sampling_eps_min'):
      sampling_eps_min = 1e-3
      sampling_eps_max = 1.0
    (input_tokens, output_tokens,
     attention_mask) = self._maybe_sub_sample(
       x0, attention_mask)
    # self.parameterization == 'subs':
    weighted_loss_map, unweighted_loss_map = self._forward_pass_diffusion(
      input_tokens,
      sampling_eps_min=sampling_eps_min,
      sampling_eps_max=sampling_eps_max,)
    
    if not self.training:
      attention_mask[:, 0] = 0
    
    # 计算 Weighted Mean (NLL)
    nlls = (weighted_loss_map * attention_mask)
    token_nll = nlls.sum() / attention_mask.sum()
    # 计算 Unweighted Mean
    unweighted_nlls_scalar = (unweighted_loss_map * attention_mask).sum() / attention_mask.sum()
    
    return Loss(loss=token_nll,
                nlls=nlls,
                token_mask=attention_mask,
                unweighted_loss=unweighted_nlls_scalar)

  def _maybe_sub_sample(self, x0, attention_mask):
    seqlen = x0.shape[1]
    if seqlen > self.num_tokens:
      assert seqlen == 2 * self.num_tokens
      # cropping is needed for text8-crop dataset
      # try the same starting point for now
      start = np.random.choice(self.num_tokens)
      end = start + self.num_tokens
      input_tokens = x0[:, start: end]
      output_tokens = x0[:, start + 1: end + 1]
      new_attention_mask = attention_mask[:, start: end]

    else:
      input_tokens = x0
      output_tokens = None
      new_attention_mask = attention_mask
    
    return input_tokens, output_tokens, new_attention_mask

  def _forward_pass_diffusion(self, x0, t=None, sampling_eps_min=None, sampling_eps_max=None):
    if t is None:
      t = self._sample_t(x0.shape,
                         x0.device,
                         sampling_eps_min,
                         sampling_eps_max)

    loss_scale, p = self.noise(t)
    sigma = self._sigma_from_p(p[:,0].unsqueeze(-1))

    xt = self.q_xt(x0,
                   p,
                   sampling_eps_min=sampling_eps_min,
                   sampling_eps_max=sampling_eps_max)
    if sampling_eps_min is not None and sampling_eps_min > 0.5:
      loss_scale = - torch.ones_like(loss_scale)
      
    # 从全 MASK 序列开始预测，无需 bos
    # xt[:, 0] = x0[:, 0]  # 保留起始 token 不被加噪
    
    x_input = xt
    x_input = torch.cat((xt, x0), dim=-1)

    model_output = self.forward(x_input, sigma=sigma)
    utils.print_nans(model_output, 'model_output')

    # 修改：Loss 计算适配 256 维 logits 和 Mapping
    # 1. 准备 Targets: 将 x0 (Token IDs) 映射为 0-255
    target_pixel_indices = self.vocab_map[x0] # [B, L]
    
    # 2. 处理非像素 Token (如 x0 中可能包含非 0-255 的特殊 token)
    # 创建一个用于 gather 的 safe indices (将 -1 替换为 0，随后用 mask 过滤)
    gather_indices = target_pixel_indices.clone()
    valid_target_mask = (target_pixel_indices != -1).float()
    gather_indices[gather_indices == -1] = 0 
    
    # 3. Gather Log Probs
    log_p_theta = torch.gather(
      input=model_output,
      dim=-1,
      index=gather_indices[:, :, None]).squeeze(-1)
    
    # 4. 过滤无效 Target 的 Loss
    log_p_theta = log_p_theta * valid_target_mask
    
    weighted_loss = loss_scale * log_p_theta  # weighted cross entropy
    unweighted_loss = -log_p_theta
    
    return weighted_loss, unweighted_loss

  def _sample_t(
      self, batch_dims, device, sampling_eps_min, sampling_eps_max, block_size=None):
    if block_size is None:
      block_size = self.block_size
    n = batch_dims[-1]
    num_blocks = n // block_size
    _eps_b = torch.rand((batch_dims[0], num_blocks), device=device)

    # antithetic sampling along blocks & batches (for uniform sampling)
    offset_b = torch.arange(batch_dims[0] * num_blocks, device=device) / (batch_dims[0] * num_blocks)
    offset_b = offset_b.view(batch_dims[0], num_blocks)
    _eps_b = (_eps_b / (batch_dims[0] * num_blocks) + offset_b) % 1
    t = _eps_b
    if block_size != self.config.model.length:
      t = t.repeat_interleave(block_size, dim=-1)

    # nll
    if sampling_eps_max >= 1 and sampling_eps_min >= 1:
      return torch.ones_like(t)
    t = t * (sampling_eps_max - sampling_eps_min) + sampling_eps_min
    return t

  def _sigma_from_p(self, p):
    return torch.min(- torch.log(1 - p), self.noise.sigma_max)

  def q_xt(
      self, x, p, block_size=None, sampling_eps_min=None, sampling_eps_max=None):
    """Computes the noisy sample xt.

    Args:
      x: int torch.Tensor with shape (batch_size,
          diffusion_model_input_length), input. 
      p: float torch.Tensor with shape (batch_size, 1).
      block_size: int, block size.
      sampling_eps_min: float, minimum percentage of masked tokens.
      sampling_eps_max: float, maximum percentage of masked tokens.
    """
    if block_size is None:
      block_size = self.block_size
  
    move_indices = torch.rand(
      * x.shape, device=x.device) <= p
    xt = torch.where(move_indices, self.mask_index, x)

    # no need to resample for bounds 1e-3, 1
    if self.config.training.resample and \
      not (sampling_eps_min == 1e-3 and sampling_eps_max == 1.0):
      xt = xt.reshape(xt.shape[0], -1, block_size)
      xt = self._resample_q_xt(x,
                               xt,
                               move_indices,
                               p,
                               block_size,
                               sampling_eps_min,
                               sampling_eps_max)
      xt = xt.reshape(xt.shape[0], -1)
    return xt

  def _resample_q_xt(
      self, x, xt, move_indices, p, block_size, sampling_eps_min, sampling_eps_max):
    """Resamples x_t if the percentage of masked tokens is outside the bounds
    defined by sampling_eps_min and sampling_eps_max."""
    perc_masked = (xt == self.mask_index).float().sum(-1) / block_size
    while (perc_masked < sampling_eps_min).any() or \
      (perc_masked > sampling_eps_max).any():
      # if a bound is epsilon, don't resample
      if sampling_eps_min == 1e-3 and sampling_eps_max != 1:
        regen_idx = (perc_masked > sampling_eps_max)
        if regen_idx.max() == 0:
          break
      elif sampling_eps_min != 1e-3 and sampling_eps_max == 1:
        regen_idx = (perc_masked < sampling_eps_min)
        if regen_idx.max() == 0:
          break
      elif sampling_eps_min != 1e-3 and sampling_eps_max != 1:
        regen_idx = (perc_masked < sampling_eps_min) | (perc_masked > sampling_eps_max)
      regen_idx = regen_idx.repeat_interleave(block_size,dim=-1)
      move_indices[regen_idx] = (torch.rand(
        * x.shape, device=x.device) < p)[regen_idx]
      xt = torch.where(move_indices, self.mask_index, x)
      xt = xt.reshape(xt.shape[0], -1, block_size)
      perc_masked = (xt == self.mask_index).float().sum(-1) / block_size
    return xt

  # def _clipped_schedule_search(self):
  #   # collect losses per batch across devices and sum them per interval
  #   best_var = float('inf')
  #   for (eps_min, eps_max), var in self.metrics.valid_vars.items():
  #     all_vars = torch.tensor(0., device=self.device)
  #     for i in range(len(var)):
  #       agg_var = var[i].to(self.device)
  #       agg_var = self.all_gather(agg_var)
  #       all_vars += agg_var.var()
  #     if all_vars < best_var:
  #       best_var = all_vars
  #       sampling_eps_min_best = eps_min
  #       sampling_eps_max_best = eps_max
  #     self.log(f'valid_var_{round(eps_min, 2)} - {round(eps_max, 2)}',
  #               all_vars / len(var),
  #               on_epoch=True,
  #               on_step=False,
  #               sync_dist=True)
  #   self.sampling_eps_min.fill_(sampling_eps_min_best)
  #   self.sampling_eps_max.fill_(sampling_eps_max_best)

  def _clipped_schedule_search(self):
    # collect losses per batch across devices and sum them per interval
    best_var = float('inf')
    
    sampling_eps_min_best = self.sampling_eps_min.item()
    sampling_eps_max_best = self.sampling_eps_max.item()

    for (eps_min, eps_max), var_list in self.metrics.valid_vars.items():
        # var_list 是一个 list，长度为 batch_num，里面是 Tensor
        # 1. 【本地合并】：先将 list 转为 Tensor [Batch_Size]
        local_vars = torch.cat(var_list, dim=0).to(self.device)
        
        # 2. 【一次通信】：一次性把所有 GPU 的所有 Batch 数据拿过来
        global_vars = self.all_gather(local_vars) 
        
        # 3. 【集中计算】：在拿到所有数据后，一次性计算方差
        # 计算总方差，对应原逻辑：对每个 Batch (跨 GPU 收集后) 算方差，然后求和
        # [World_Size, Num_Batches, Batch_items] -> permute -> [Num_Batches, World_Size, Batch_items]
        # -> flatten(1) -> [Num_Batches, Total_Items_In_Batch]
        if global_vars.dim() == 3:
             # 假设 local_vars 是 [Num_Batches, Items]
             combined = global_vars.permute(1, 0, 2).reshape(len(var_list), -1)
             current_total_var = combined.var(dim=1).sum()
        else:
             # 无法确定维度，还是在本地循环计算
             current_total_var = 0.0
             # global_vars: [World_Size, Num_Batches, ...]
             # 按 Batch 维度切分
             num_batches = local_vars.shape[0]
             for i in range(num_batches):
                 # 取出第 i 个 batch 在所有 GPU 上的数据
                 batch_data = global_vars[:, i, ...] 
                 current_total_var += batch_data.var()

        if current_total_var < best_var:
            best_var = current_total_var
            sampling_eps_min_best = eps_min
            sampling_eps_max_best = eps_max
            
        self.log(f'valid_var_{round(eps_min, 2)} - {round(eps_max, 2)}',
                 current_total_var / len(var_list), # 原逻辑是除以 batch 数
                 on_epoch=True,
                 on_step=False,
                 sync_dist=True)

    self.sampling_eps_min.fill_(sampling_eps_min_best)
    self.sampling_eps_max.fill_(sampling_eps_max_best)
  
  def _compute_entropy(self, x):
    _, counts = torch.unique(x, return_counts=True, sorted=False)
    entropy = torch.special.entr(counts.float() / counts.sum()).sum()
    return entropy
