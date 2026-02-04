import os
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ["TOKENIZERS_PARALLELISM"] = "false"
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
from torch.optim.lr_scheduler import LambdaLR

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
      
      # ================== 核心修复开始 ==================
      # [说明] 无论预训练模型是什么配置，都强制使用当前 YAML 的参数覆盖它
      # 避免 block_size 错位或 cross_attn 未开启导致的数据泄露
      print(f"[Force Config] Overriding model config: block_size={self.config.block_size}, cross_attn=True")
      # 1. 强制更新 Config 对象中的关键参数
      self.backbone.config.model_length = self.config.model.length
      self.backbone.config.block_size = self.config.block_size  # 关键：修正为 256
      self.backbone.config.cross_attn = True                    # 关键：开启 Cross Attn
      self.backbone.config.attn_backend = self.config.model.attn_backend
      
      # 2. 强制将配置同步到底层 Backbone 模块 (防止有模块没引用 config 对象)
      if hasattr(self.backbone, "backbone"):
        # 同步属性
        self.backbone.backbone.block_size = self.config.block_size
        self.backbone.backbone.cross_attn = True
        self.backbone.backbone.attn_backend = self.config.model.attn_backend
        
        # 遍历所有 Block，确保 Attention 后端正确 (替代了原代码的 for 循环功能)
        if hasattr(self.backbone.backbone, "blocks"):
          for block in self.backbone.backbone.blocks:
            block.attn_backend = self.config.model.attn_backend
            block.block_size = self.config.block_size # 确保 Block 内部切分逻辑正确
        
        # 3. [最关键一步] 重新生成 Attention Mask
        # 这会根据 cross_attn=True 和 block_size=256 生成正确的掩码
        # 从而遮挡住 x0，防止泄露
        print(f"[Mask Gen] Regenerating attention mask with block_size={self.config.block_size}...")
        self.backbone.backbone.gen_mask(
          seqlen=self.config.model.length, 
          block_size=self.config.block_size, 
          attn_backend=self.config.model.attn_backend
        )
      # ================== 核心修复结束 ==================
      
      # Regenerate mask if pretrained model uses flex attention mask
      # and current model uses sdpa mask
      # if getattr(self.backbone.config, 'attn_backend', None) == 'flex' and \
      #   self.config.model.attn_backend == 'sdpa':
      #   self.backbone.config.attn_backend = 'sdpa'
      #   for i in self.backbone.backbone.blocks:
      #     i.attn_backend = 'sdpa'
      #   self.backbone.backbone.gen_mask(self.config.model.length, self.block_size, attn_backend='sdpa')
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
        ids = tokenizer.encode(str(i), add_special_tokens=False)
        assert len(ids) == 1, f"Error: Number {i} tokenizes to {ids}"
        pixel_ids_list.append(ids[0])
        # pixel_ids_list.append(self.tokenizer.convert_tokens_to_ids(str(i)))
    
    # 注册 buffer，会自动随模型移动到 GPU
    self.register_buffer('pixel_token_ids', torch.tensor(pixel_ids_list, dtype=torch.long))
    
    # 创建全词表到 0-255 的映射表 (默认为 -1)
    vocab_map_tensor = torch.full((self.vocab_size,), -1, dtype=torch.long)
    for i, pid in enumerate(pixel_ids_list):
        if pid < self.vocab_size:
            vocab_map_tensor[pid] = i
    self.register_buffer('vocab_map', vocab_map_tensor)
    
    # ================== 改进方案 1: Embedding 重构 ==================
    # 目的：让 token "128" 和 "129" 的向量由于数学特性天然相似
    # 获取模型原本的 embedding 层引用 (根据 modeling_bd3lm.py 结构)
    # BD3LM -> DITBackbone -> EmbeddingLayer -> nn.Parameter
    embedding_layer = self.backbone.backbone.vocab_embed.embedding 
    hidden_dim = embedding_layer.shape[1]

    # 创建正弦编码 (Sinusoidal Position Encoding 风格)，[256, hidden_dim]
    position = torch.arange(256).unsqueeze(1).float()
    div_term = torch.exp(torch.arange(0, hidden_dim, 2).float() * -(math.log(10000.0) / hidden_dim))
    sinusoidal_embed = torch.zeros(256, hidden_dim)
    sinusoidal_embed[:, 0::2] = torch.sin(position * div_term)
    sinusoidal_embed[:, 1::2] = torch.cos(position * div_term)

    # 强制覆盖预训练模型中对应的 Token Embedding
    # 注意：这会破坏该 Token 原本的文本语义，但对于纯像素压缩任务是正向的
    with torch.no_grad():
        ids = self.pixel_token_ids.to(embedding_layer.device) 
        sinusoidal_embed = sinusoidal_embed.to(embedding_layer.device)
        
        # 1. 覆盖 Input Embedding
        embedding_layer[ids] = sinusoidal_embed
        
        # 2. 同步 Output Layer 权重
        output_weight = self.backbone.backbone.output_layer.linear.weight
        # 检查是否已经是共享权重 (Weight Tying)
        if embedding_layer.data_ptr() == output_weight.data_ptr():
          print("[Info] Weights are tied. Output layer is already updated.")
        else:
          print("[Optimization] Syncing output layer weights with Sinusoidal embeddings...")
          # 确保形状匹配 (通常都是 [50257, Hidden])
          if output_weight.shape == embedding_layer.shape:
            output_weight[ids] = embedding_layer[ids]
          else:
            # 防止形状转置的情况 (虽然在HF GPT2实现中很少见，但为了稳健)
            output_weight[ids, :] = embedding_layer[ids, :]
          print("[Optimization] Output layer synced.")

    print("[Optimization] Re-initialized Pixel Embeddings & Output Weights with Sinusoidal structure.")
    # =============================================================


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
    if not self.config.algo.fix_clipping and not self.trainer.sanity_checking:
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
    # 2. BPP (Bits Per Pixel)
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
    # 1. ------------- 差分学习率策略 (Differential LR) -------------
    # 将参数分为两组：Backbone (预训练部分) 和 New Layers (新初始化的Embedding/Head)
    # 假设你的配置文件中有 self.config.optim.lr 作为基础学习率
    base_lr = self.config.optim.lr
    backbone_lr_ratio = 0.3  # Backbone 学习率是基础学习率的 0.3 倍
    
    backbone_params = []
    new_params = []
    
    # 关键词匹配：根据实际打印出的 named_parameters 调整这里的关键词
    modules = [name for name, _ in self.named_parameters()]
    target_new_modules = ['vocab_embed.embedding', 'output_layer.linear']
    
    for name, param in self.named_parameters():
      if not param.requires_grad:
        continue
      
      # 检查参数名是否包含新层的关键词
      is_new = any(k in name for k in target_new_modules)
      
      if is_new:
        # print(f"  -> High LR (New): {name}")
        new_params.append(param)
      else:
        backbone_params.append(param)
    # print(f"[Optimizer] Summary: {len(new_params)} params in New Group (High LR), {len(backbone_params)} params in Backbone Group (Low LR).\n")
    # 创建参数组
    optimizer_groups = [
      # 组1：Backbone，使用较小的学习率
      {'params': backbone_params, 'lr': base_lr * backbone_lr_ratio},
      # 组2：新层，使用全速学习率
      {'params': new_params, 'lr': base_lr}
    ]

    optimizer = torch.optim.AdamW(
      optimizer_groups, # 传入参数组而非全部参数
      betas=(self.config.optim.beta1, self.config.optim.beta2),
      eps=self.config.optim.eps,
      weight_decay=self.config.optim.weight_decay
    )

    # 2. ------------- 自定义调度器 (Warmup -> Constant -> Decay) -------------
    # 我们放弃 hydra 的自动实例化，改用 LambdaLR 手写逻辑
    # 假设总步数从 trainer 获取 (如果不方便获取，可以在 config 里硬编码 max_steps)
    if self.trainer.max_steps and self.trainer.max_steps > 0:
      total_steps = self.trainer.max_steps
    else:
      total_steps = self.config.training.max_steps # 备选方案，取决于你的config结构

    # 定义阶段比例
    warmup_ratio = 0.1  # 前 10% Warmup
    constant_ratio = 0.3 # 接下来 30% 保持 Constant
    # 剩余为 Decay
    
    warmup_steps = int(total_steps * warmup_ratio)
    constant_end_step = int(total_steps * (warmup_ratio + constant_ratio))

    def lr_lambda(current_step):
      # 阶段 1: Warmup
      if current_step < warmup_steps:
        return float(current_step) / float(max(1, warmup_steps))
      
      # 阶段 2: Constant (保持 1.0 倍率)
      elif current_step < constant_end_step:
        return 1.0
      
      # 阶段 3: Decay (Cosine Decay 到 0)
      else:
        progress = float(current_step - constant_end_step) / float(max(1, total_steps - constant_end_step))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

    scheduler_dict = {
        'scheduler': scheduler,
        'interval': 'step',
        'monitor': 'val/loss',
        'name': 'trainer/lr'
    }
    
    return [optimizer], [scheduler_dict]

  # def configure_optimizers(self):
  #   # TODO(yair): Lightning currently giving this warning when using `fp16`:
  #   #  "Detected call of `lr_scheduler.step()` before `optimizer.step()`. "
  #   #  Not clear if this is a problem or not.
  #   #  See: https://github.com/Lightning-AI/pytorch-lightning/issues/5558
  #   optimizer = torch.optim.AdamW(
  #     self._get_parameters(),
  #     lr=self.config.optim.lr,
  #     betas=(self.config.optim.beta1,
  #            self.config.optim.beta2),
  #     eps=self.config.optim.eps,
  #     weight_decay=self.config.optim.weight_decay)

  #   scheduler = hydra.utils.instantiate(
  #     self.config.lr_scheduler, optimizer=optimizer)
  #   scheduler_dict = {'scheduler': scheduler,
  #                     'interval': 'step',
  #                     'monitor': 'val/loss',
  #                     'name': 'trainer/lr'}
  #   return [optimizer], [scheduler_dict]

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

  def _get_current_sigma(self):
    # 初始 sigma (比较模糊，利于收敛)
    start_sigma = 2.0
    # 最终 sigma (比较锐利，利于细节)
    end_sigma = 0.5
    # 获取当前步数和总步数
    step = self.trainer.global_step
    
    if hasattr(self, 'trainer') and self.trainer.max_steps > 0:
        total_steps = self.trainer.max_steps
    else:
        total_steps = 60000 # 兜底默认值
        
    # 计算进度 (0.0 -> 1.0)
    progress = min(1.0, step / total_steps)
    
    # 策略：线性衰减 (Linear Decay)
    current_sigma = start_sigma - (start_sigma - end_sigma) * progress
    
    return current_sigma

  def _forward_pass_diffusion(self, x0, t=None, sampling_eps_min=None, sampling_eps_max=None):
    # ================= 1. 采样噪声与时间步 =================
    if t is None:
      t = self._sample_t(x0.shape,
                         x0.device,
                         sampling_eps_min,
                         sampling_eps_max)

    loss_scale, p = self.noise(t)
    
    # # === [Debug] ===
    # if self.global_step % 10 == 0:
    #     print(f"\n[Debug] Step={self.global_step}")
    #     print(f"  Max Loss Scale: {loss_scale.max().item():.4f}")
    #     print(f"  Min Loss Scale: {loss_scale.min().item():.4f}")
    # # ===============
    
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
    # # === [Debug] ===
    # if self.global_step % 10 == 0:
    #     print(f"  Logits Max: {model_output.max().item():.4f}")
    #     print(f"  Logits Min: {model_output.min().item():.4f}")
    # # ===============
    utils.print_nans(model_output, 'model_output')

    # ================= 3. Soft Loss 计算 (核心修改部分) =================
    # 3.1 准备目标数据
    # 将 x0 (Token ID) 映射回 0-255 的像素值索引
    target_pixel_indices = self.vocab_map[x0] # 形状 [B, L]
    
    # 3.2 处理无效目标 (Mask)
    # 有些 token 可能不是像素 (比如 padding 或特殊符号)，它们在 vocab_map 中是 -1
    # 我们先生成一个 mask，并在计算前把 -1 替换成 0 以免报错
    valid_target_mask = (target_pixel_indices != -1).float()
    safe_targets = target_pixel_indices.clone()
    safe_targets[safe_targets == -1] = 0 # 临时填0，计算完后会用 mask 乘掉
    
    # ----------------- 计算 Unweighted Loss (Hard NLL) -----------------
    # 我们保留一份“原始”的 Loss，用于您在日志里看 BPP 指标
    # 这代表了模型预测“准确数值”的能力，不包含平滑
    log_p_hard = torch.gather(
      input=model_output,
      dim=-1,
      index=safe_targets.unsqueeze(-1)).squeeze(-1)
    
    unweighted_loss = log_p_hard * valid_target_mask

    # ------------- 计算 Weighted Loss (Soft Cross Entropy) ------------
    # [参数设置] 平滑的 sigma。
    # sigma=0.5 比较尖锐（接近原始），sigma=2.0 比较平滑（容忍邻居）
    smoothing_sigma = self._get_current_sigma()
    self.log('train/sigma', smoothing_sigma, on_step=True, on_epoch=False, prog_bar=True)
    
    # 创建 [0, 1, ..., 255] 的向量用于广播计算距离
    pixel_range = torch.arange(256, device=model_output.device).view(1, 1, 256)
    
    # 扩展目标维度以便广播: [B, L, 1]
    target_expand = safe_targets.unsqueeze(-1)
    
    # 计算高斯分布权重: exp( - (x - target)^2 / (2 * sigma^2) )
    dist_sq = (pixel_range - target_expand) ** 2
    soft_weights = torch.exp(-dist_sq / (2 * smoothing_sigma ** 2))
    
    # 归一化，使其成为一个合法的概率分布 (和为1)，“软目标” Q(x)
    soft_targets = soft_weights / soft_weights.sum(dim=-1, keepdim=True)
    
    # 计算 Soft Cross Entropy: - Sum( Q(x) * LogP(x) )
    # model_output 已经是 LogP(x)，计算了预测分布 P 和软目标 Q 之间的交叉熵
    soft_ce = -(soft_targets * model_output).sum(dim=-1)
    
    # # 应用 Mask 和 Diffusion 的 Loss Scale
    # weighted_loss = loss_scale * (-soft_ce * valid_target_mask)
    
    # 核心修复：只对被 Mask 的 Token 计算 Loss
    # 1. 找出当前时间步 t 中，哪些位置是被 Mask 的 (需要预测的)
    # xt 是加噪后的序列，self.mask_index 是 mask token 的 ID
    is_masked = (xt == self.mask_index).float()
    
    # 2. 组合 Mask：(有效像素) AND (被 Mask 掉)
    # valid_target_mask 负责过滤 Padding
    # is_masked 负责过滤掉那些未被加噪、已知的部分 (避免 Soft Loss 冲突)
    final_loss_mask = valid_target_mask * is_masked
    
    # 3. 应用 Loss Scale 和 Mask
    weighted_loss = loss_scale * (-soft_ce * final_loss_mask)
    
    # # === [DEBUG] ===
    # if self.global_step % 10 == 0:
    #     # 计算 Mask 后的真实平均 Loss (只看被 mask 的部分)
    #     masked_loss_sum = weighted_loss.sum()
    #     mask_count = final_loss_mask.sum()
    #     real_avg_loss = masked_loss_sum / (mask_count + 1e-6)
        
    #     print(f"\n[Debug] Step={self.global_step}")
    #     print(f"  Raw Soft CE Max: {soft_ce.max().item():.2f} (Huge is expected!)")
    #     print(f"  Masked Loss Sum: {masked_loss_sum.item():.4f}")
    #     print(f"  Real Avg Loss:   {real_avg_loss.item():.4f} (Should be small, e.g. < 20)")
    # # ===============
    
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
    
    # [Fix] 增加计数器，防止死循环
    cnt = 0 
    max_iter = 50
    
    while (perc_masked < sampling_eps_min).any() or \
      (perc_masked > sampling_eps_max).any():
      # [Fix] 超过次数强制退出循环
      if cnt > max_iter: break
      cnt += 1
      
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

    if self.config.algo.fix_clipping == False:
      self.sampling_eps_min.fill_(sampling_eps_min_best)
      self.sampling_eps_max.fill_(sampling_eps_max_best)
  
  def _compute_entropy(self, x):
    _, counts = torch.unique(x, return_counts=True, sorted=False)
    entropy = torch.special.entr(counts.float() / counts.sum()).sum()
    return entropy
