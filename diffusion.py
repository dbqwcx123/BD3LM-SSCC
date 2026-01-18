import os
os.environ['TRANSFORMERS_OFFLINE'] = '1'
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

import dataloader
import metrics
import models
import noise_schedule
import utils

def _sample_categorical(categorical_probs):
  gumbel_norm = (1e-10 - (torch.rand_like(categorical_probs) + 1e-10).log())
  samples = (categorical_probs / gumbel_norm).argmax(dim=-1)
  return samples

@dataclass
class Loss:
  loss: torch.FloatTensor
  nlls: torch.FloatTensor
  token_mask: torch.FloatTensor


class Diffusion(L.LightningModule):
  def __init__(
    self,
    config,
    tokenizer):
    super().__init__()
    self.save_hyperparameters()
    self.config = config
    self.tokenizer = tokenizer
    self.vocab_size = self.tokenizer.vocab_size
    self.sampler = self.config.algo.sampler
    self.antithetic_sampling = self.config.training.antithetic_sampling
    self.cross_attn = self.config.algo.cross_attn
    self.ignore_bos = self.config.algo.ignore_bos
    if (not hasattr(self.tokenizer, 'mask_token')
        or self.tokenizer.mask_token is None):
      self.mask_index = self.vocab_size
      self.vocab_size += 1
    else:
      self.mask_index = self.tokenizer.mask_token_id
    if hasattr(self.config, 'algo'):
      self.parameterization = self.config.algo.parameterization
    else:
      self.parameterization = self.config.parameterization
    if hasattr(self.config, 'block_size'):
      self.block_size = self.config.block_size
    else:
      self.block_size = self.config.model.length
    if self.config.algo.backbone == 'dit':
      self.backbone = models.dit.DIT(
        self.config, vocab_size=self.vocab_size)
    elif self.config.algo.backbone == 'hf_dit':
      self.backbone = transformers.AutoModelForMaskedLM.from_pretrained(
        config.eval.checkpoint_path, trust_remote_code=True, local_files_only=True)
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

    if self.config.training.ema > 0:
      self.ema = models.ema.ExponentialMovingAverage(
        self._get_parameters(),
        decay=self.config.training.ema)
    else:
      self.ema = None
    
    self.var_min = self.config.algo.var_min
    if self.var_min:
      self.register_buffer('sampling_eps_min', torch.tensor(
        self.config.training.sampling_eps_min))
      self.register_buffer('sampling_eps_max', torch.tensor(
        self.config.training.sampling_eps_max))
      
    self.time_conditioning = self.config.algo.time_conditioning
    self.neg_infinity = -1000000.0
    self.fast_forward_epochs = None
    self.fast_forward_batches = None
    self._validate_configuration()

  def _get_parameters(self):
    parameters = [self.backbone.parameters(),
                  self.noise.parameters()]
    return itertools.chain(* parameters)

  def on_validation_model_zero_grad(self) -> None:
    '''
    Small hack to avoid first validation on resume. 
    This will NOT work if the gradient accumulation step should be performed at this point.
    '''
    super().on_validation_model_zero_grad()
    if self.trainer.ckpt_path is not None and getattr(self, '_restarting_skip_val_flag', True):
        self.trainer.sanity_checking = True
        self._restarting_skip_val_flag = False

  def _validate_configuration(self):
    if self.config.mode == 'sample_eval' and \
        self.config.sampling.first_hitting:
      assert self.config.loader.eval_batch_size == 1
    assert self.config.algo.backbone in {'dit', 'ar', 'hf_dit'}
    if self.config.sampling.kv_cache:
      assert self.config.algo.name in {'ar', 'bd3lm'}
    
    if self.config.mode == 'sample_eval':
      assert self.config.model.attn_backend != 'flex', 'FlexAttention mask not supported at inference.'
    if self.config.model.attn_backend == 'flex':
      assert self.config.algo.name == 'bd3lm', 'Custom FlexAttention mask only supported for BD3LM.'
      
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

  def _replace_ckpt_keys(self, checkpoint):
    state_dict = checkpoint['state_dict']
    new_state_dict = OrderedDict()
    for k,v in state_dict.items():
      new_state_dict[k.replace('_orig_mod.', '')] = v
    checkpoint['state_dict'] = new_state_dict
    return checkpoint

  def on_load_checkpoint(self, checkpoint):
    print('Loading checkpoint at', checkpoint['global_step'])
    self._restarting_skip_val_flag = True

    # for models compiled with `torch.compile`
    if '_orig_mod.' in list(checkpoint['state_dict'].keys())[0]:
      checkpoint = self._replace_ckpt_keys(checkpoint)

    if self.ema:
      self.ema.load_state_dict(checkpoint['ema'])
    if 'sampling_eps_min' in checkpoint.keys():
      self.sampling_eps_min = checkpoint['sampling_eps_min']
      self.sampling_eps_max = checkpoint['sampling_eps_max']
    # Copied from:
    # https://github.com/Dao-AILab/flash-attention/blob/main/training/src/datamodules/language_modeling_hf.py#L41
    self.fast_forward_epochs = checkpoint['loops'][
      'fit_loop']['epoch_progress']['current']['completed']
    self.fast_forward_batches = checkpoint['loops'][
      'fit_loop']['epoch_loop.batch_progress'][
        'current']['completed']

  def on_save_checkpoint(self, checkpoint):
    if self.ema:
      checkpoint['ema'] = self.ema.state_dict()
    if hasattr(self, 'sampling_eps_min'):
      checkpoint['sampling_eps_min'] = self.sampling_eps_min
      checkpoint['sampling_eps_max'] = self.sampling_eps_max
    # Copied from:
    # https://github.com/Dao-AILab/flash-attention/blob/main/training/src/tasks/seq.py
    # ['epoch_loop.batch_progress']['total']['completed'] is 1 iteration
    # behind, so we're using the optimizer's progress.
    checkpoint['loops']['fit_loop'][
      'epoch_loop.batch_progress']['total'][
        'completed'] = checkpoint['loops']['fit_loop'][
          'epoch_loop.automatic_optimization.optim_progress'][
            'optimizer']['step']['total'][
              'completed'] * self.trainer.accumulate_grad_batches
    checkpoint['loops']['fit_loop'][
      'epoch_loop.batch_progress']['current'][
        'completed'] = checkpoint['loops']['fit_loop'][
          'epoch_loop.automatic_optimization.optim_progress'][
            'optimizer']['step']['current'][
              'completed'] * self.trainer.accumulate_grad_batches
    # _batches_that_stepped tracks the number of global steps, not the number
    # of local steps, so we don't multiply with self.trainer.accumulate_grad_batches here.
    checkpoint['loops']['fit_loop'][
      'epoch_loop.state_dict'][
        '_batches_that_stepped'] = checkpoint['loops']['fit_loop'][
          'epoch_loop.automatic_optimization.optim_progress'][
            'optimizer']['step']['total']['completed']
    if 'sampler' not in checkpoint.keys():
      checkpoint['sampler'] = {}
    if hasattr(self.trainer.train_dataloader.sampler,
               'state_dict'):
      sampler_state_dict = self.trainer.\
        train_dataloader.sampler.state_dict()
      checkpoint['sampler'][
        'random_state'] = sampler_state_dict.get(
          'random_state', None)
    else:
      checkpoint['sampler']['random_state'] = None

  def on_train_start(self):
    if self.ema:
      self.ema.move_shadow_params_to_device(self.device)
    # Adapted from:
    # https://github.com/Dao-AILab/flash-attention/blob/main/training/src/datamodules/language_modeling_hf.py
    distributed = (
      self.trainer._accelerator_connector.use_distributed_sampler
      and self.trainer._accelerator_connector.is_distributed)
    if distributed:
      sampler_cls = dataloader.FaultTolerantDistributedSampler
    else:
      sampler_cls = dataloader.RandomFaultTolerantSampler
    updated_dls = []
    for dl in self.trainer.fit_loop._combined_loader.flattened:
      if hasattr(dl.sampler, 'shuffle'):
        dl_sampler = sampler_cls(
          dl.dataset, shuffle=dl.sampler.shuffle)
      else:
        dl_sampler = sampler_cls(dl.dataset)
      if (distributed
          and self.fast_forward_epochs is not None
          and self.fast_forward_batches is not None):
        dl_sampler.load_state_dict({
          'epoch': self.fast_forward_epochs,
          'counter': (self.fast_forward_batches
                      * self.config.loader.batch_size)})
      updated_dls.append(
        torch.utils.data.DataLoader(
          dl.dataset,
          batch_size=self.config.loader.batch_size,
          num_workers=self.config.loader.num_workers,
          pin_memory=self.config.loader.pin_memory,
          sampler=dl_sampler,
          shuffle=False,
          persistent_workers=True))
    self.trainer.fit_loop._combined_loader.flattened = updated_dls

  def optimizer_step(self, *args, **kwargs):
    super().optimizer_step(*args, **kwargs)
    if self.ema:
      self.ema.update(self._get_parameters())

  def _subs_parameterization(self, logits, xt):
    # log prob at the mask index = - infinity
    logits[:, :, self.mask_index] += self.neg_infinity
    
    # Normalize the logits such that x.exp() is
    # a probability distribution over vocab_size.
    logits = logits - torch.logsumexp(logits, dim=-1,
                                      keepdim=True)
    
    # Apply updates directly in the logits matrix.
    # For the logits of the unmasked tokens, set all values
    # to -infinity except for the indices corresponding to
    # the unmasked tokens.
    unmasked_indices = (xt != self.mask_index)
    logits[unmasked_indices] = self.neg_infinity
    logits[unmasked_indices, xt[unmasked_indices]] = 0
    return logits

  def _process_sigma(self, sigma):
    # cause of overfitting for block size 1?
    assert sigma.ndim == 2
    sigma = sigma.mean(-1).squeeze()
    if sigma.ndim == 0:
      sigma = sigma.unsqueeze(0)
    if not self.time_conditioning:
      sigma = torch.zeros_like(sigma)
    assert sigma.ndim == 1, sigma.shape
    return sigma

  def forward(self, x, sigma, sample_mode=False, store_kv=False):
    """Returns log score."""
    sigma = self._process_sigma(sigma)
    with torch.amp.autocast('cuda', dtype=torch.float32):
      logits = self.backbone(x, sigma, sample_mode, store_kv)  # [Modified]

    if self.cross_attn:
      x = x[:, :self.config.model.length]
    # self.parameterization == 'subs':
    return self._subs_parameterization(logits=logits, xt=x)
    # return logits
    
  def on_train_epoch_start(self):
    self.backbone.train()
    self.noise.train()
    self.metrics.reset()
    assert self.metrics.train_nlls.nll.mean_value == 0
    assert self.metrics.train_nlls.nll.weight == 0

  def training_step(self, batch, batch_idx):
    del batch_idx
    losses = self._loss(batch['input_ids'],
                        batch['attention_mask'])
    self.metrics.train_nlls.update(losses.nlls, losses.token_mask)
    self.log(name='trainer/loss',
             value=losses.loss.item(),
             on_step=True,
             on_epoch=False,
             sync_dist=True)
    return losses.loss

  def on_validation_epoch_start(self):
    self.metrics.reset()
    if self.ema:
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
    if self.ema:
      self.ema.restore(self._get_parameters())
    if self.var_min and not self.trainer.sanity_checking:
      self._clipped_schedule_search()
      self.log('sampling_eps_min',
               self.sampling_eps_min,
               on_epoch=True,
               on_step=False,
               sync_dist=True)
      self.log('sampling_eps_max',
               self.sampling_eps_max,
               on_epoch=True,
               on_step=False,
               sync_dist=True)
  
  def _check_val_sampling_intvl(self, sampling_eps_min, sampling_eps_max):
    """Checks if the current sampling interval is valid for reporting likelihood."""
    if (sampling_eps_min == 1e-3 \
        and sampling_eps_max == 1 \
        and not (self.block_size == 1 and self.config.training.eval_nll)):
      return True # elbo
    elif (self.block_size == 1 and sampling_eps_min >= 1):
      return True # nll (block size 1)
    return False # not a valid elbo (biased estimate)
      
  def validation_step(self, batch, batch_idx):
    if self.var_min:
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
            loss=losses_clip.loss.clone())
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
    elif self.block_size == 1:
      # nll
      losses = self._loss(batch['input_ids'],
                          batch['attention_mask'],
                          sampling_eps_min=1,
                          sampling_eps_max=1)
    else:
      # nelbo
      losses = self._loss(batch['input_ids'],
                          batch['attention_mask'],
                          sampling_eps_min=1e-3,
                          sampling_eps_max=1)
    self.metrics.valid_nlls.update(losses.nlls, losses.token_mask)
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

    if block_size == 1 and sampling_eps_min == 1.0:
      return torch.full_like(x, self.mask_index)

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

  def _sample_prior(self, *batch_dims):
    return self.mask_index * torch.ones(
      * batch_dims, dtype=torch.int64, device=self.device)

  @torch.no_grad()
  def _nucleus_sample(self, p_x0):
    p = self.config.sampling.nucleus_p
    if p == 1.0:
      return p_x0
    p_x0_ = p_x0[:, -self.block_size:].clone()
    sorted_probs, sorted_indices = p_x0_.sort(dim=-1, descending=True)
    cum_probs = sorted_probs.cumsum(dim=-1)
    nucleus_mask = cum_probs <= p
    nucleus_mask[..., 0] = 1
    sorted_probs = sorted_probs * nucleus_mask
    p_x0_.scatter_(-1, sorted_indices, sorted_probs * nucleus_mask)
    p_x0_ /= p_x0_.sum(-1, keepdim=True)
    p_x0[:, -self.block_size:] = p_x0_
    return p_x0

  @torch.no_grad()
  def _ddpm_caching_update(self, x, t, dt, p_x0=None):
    _, move_chance_t = self.noise(t)
    _, move_chance_s = self.noise(t - dt)
    sigma_t = self._sigma_from_p(move_chance_t)
    move_chance_t = move_chance_t[:, None]
    move_chance_s = move_chance_s[:, None]
    mask_prob = move_chance_s / move_chance_t

    if p_x0 is None:
      if self.config.sampling.kv_cache:
        p_x0 = self.forward(x[:, -self.block_size:],
                        sigma_t,
                        sample_mode=True).to(torch.float64)
      else:   
        p_x0 = self.forward(x,
                          sigma_t,
                          sample_mode=True).to(torch.float64)
        p_x0 = p_x0[:, -self.block_size:]
      p_x0 = p_x0.exp()
      p_x0 = self._nucleus_sample(p_x0)

    if self.config.sampling.first_hitting:
      x_block = _sample_categorical(p_x0)
      # randomly and uniformly select an index in the block (among masked tokens)
      num_masked = (x[:, -self.block_size:] == self.mask_index).sum(-1)
      ind = torch.randint(0, num_masked, (x_block.shape[0],))
      ind = (x[:, -self.block_size:] == self.mask_index).nonzero()[ind, 1]
      mask = (torch.arange(self.block_size, device=x.device) == ind[:, None]).to(x_block.dtype)
      x_block = x_block * mask + x[:, -self.block_size:] * (1 - mask)
    else:
      q_xs = p_x0 * (1 - mask_prob)
      q_xs[:, :, self.mask_index] = mask_prob.squeeze(-1)
      x_block = _sample_categorical(q_xs)
    copy_flag = (x[:, -self.block_size:] != self.mask_index).to(x.dtype)
    x_block =  copy_flag * x[:, -self.block_size:] + (1 - copy_flag) * x_block
    x_new = torch.cat((x[:, :-self.block_size], x_block), dim=-1)

    # [Modify] Removed internal store_kv logic to avoid duplicate updates.
    # KV cache update is now handled in _semi_ar_sampler after the block generation is complete.
    if self.config.sampling.kv_cache and self.mask_index not in x_block:
      # compute kv cache if all tokens in a block are sampled
      _ = self.forward(x_block, sigma_t, sample_mode=True, store_kv=True)

    if not torch.allclose(x_new, x):
      return None, x_new
    else:
      return p_x0, x_new
  
  @torch.no_grad()
  def _sample(
    self, seqlen=None, num_steps=None, eps=1e-5, batch_size_per_gpu=None):
    """Generate samples from the model."""
    if seqlen is None:
      seqlen = self.config.model.length
    if batch_size_per_gpu is None:
      batch_size_per_gpu = self.config.loader.eval_batch_size
    samples = []
    # self.sampler == 'semi_ar':
    for _ in range(self.config.sampling.num_sample_batches):
      sample_i, num_tries = None, 0
      while sample_i is None:
        num_tries += 1
        # [Modify] handle non-divisible lengths
        num_strides = math.ceil(seqlen / self.block_size)
        sample_i, nfes = self._semi_ar_sampler(
          n_samples=batch_size_per_gpu,
          num_strides=num_strides,  # (seqlen // self.block_size), 
          num_steps=num_steps,
          seqlen=seqlen)
        if num_tries > 10:
          raise ValueError('Sampling failed.')
      samples.append(sample_i)
      self.metrics.nfes.update(nfes)
    samples = torch.cat(samples, dim=0) 
    return self.tokenizer.batch_decode(samples)

  def _sigma_from_p(self, p):
    return torch.min(- torch.log(1 - p), self.noise.sigma_max)

  def restore_model_and_sample(self, num_steps, eps=1e-5, seqlen=None):
    """Generate samples from the model."""
    if self.ema:  
      self.ema.store(self._get_parameters())
      self.ema.copy_to(self._get_parameters())
    self.backbone.eval()
    self.noise.eval()
    samples = self._sample(
      seqlen=seqlen,
      batch_size_per_gpu=self.config.loader.eval_batch_size,
      num_steps=num_steps,
      eps=eps)
    return samples

  def _sample_t(
      self, batch_dims, device, sampling_eps_min, sampling_eps_max, block_size=None):
    if block_size is None:
      block_size = self.block_size
    n = batch_dims[-1]
    num_blocks = n // block_size
    _eps_b = torch.rand((batch_dims[0], num_blocks), device=device)

    # antithetic sampling along blocks & batches (for uniform sampling)
    if self.antithetic_sampling:
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

      # Helps with validation ppl, since the val
      # examples will all start and end with BOS/EOS
      if self.config.data.insert_train_special == True:
        input_tokens[:, 0] = self.tokenizer.bos_token_id
        output_tokens[:, -1] = self.tokenizer.eos_token_id
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
    if self.ignore_bos:
      xt[:, 0] = x0[:, 0]
    
    x_input = xt
    if self.cross_attn:
      x_input = torch.cat((xt, x0), dim=-1)

    model_output = self.forward(x_input, sigma=sigma)
    utils.print_nans(model_output, 'model_output')

    log_p_theta = torch.gather(
      input=model_output,
      dim=-1,
      index=x0[:, :, None]).squeeze(-1)
    loss = loss_scale * log_p_theta
    return loss

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
    loss = self._forward_pass_diffusion(
      input_tokens,
      sampling_eps_min=sampling_eps_min,
      sampling_eps_max=sampling_eps_max,)
    
    if self.ignore_bos and not self.training:
      attention_mask[:, 0] = 0
      
    nlls = (loss * attention_mask)
    token_nll = nlls.sum() / attention_mask.sum()
    return Loss(loss=token_nll,
                nlls=nlls,
                token_mask=attention_mask)

  def _clipped_schedule_search(self):
    # collect losses per batch across devices and sum them per interval
    best_var = float('inf')
    for (eps_min, eps_max), var in self.metrics.valid_vars.items():
      all_vars = torch.tensor(0., device=self.device)
      for i in range(len(var)):
        agg_var = var[i].to(self.device)
        agg_var = self.all_gather(agg_var)
        all_vars += agg_var.var()
      if all_vars < best_var:
        best_var = all_vars
        sampling_eps_min_best = eps_min
        sampling_eps_max_best = eps_max
      self.log(f'valid_var_{round(eps_min, 2)} - {round(eps_max, 2)}',
                all_vars / len(var),
                on_epoch=True,
                on_step=False,
                sync_dist=True)
    if self.config.algo.fix_clipping == False:
      self.sampling_eps_min.fill_(sampling_eps_min_best)
      self.sampling_eps_max.fill_(sampling_eps_max_best)

  @torch.no_grad
  def _semi_ar_sampler(
    self, n_samples, num_steps, num_strides, seqlen, context_size=1024):
    if seqlen is None:
      seqlen = self.config.model.length
    sampling_steps = 0
    

    ones = torch.ones((n_samples,1), dtype=self.dtype,
                      device=self.device)
    
    # reset kvs
    if self.config.sampling.kv_cache:
      self.backbone.reset_kv_cache(eval_batch_size=self.config.loader.eval_batch_size)

    for stride_num in tqdm(range(num_strides)):
      # sample next block
      if stride_num == 0:
        x_accum = self._sample_prior(n_samples, self.block_size).to(self.device)
        x_accum[:, 0] = self.tokenizer.bos_token_id
      else:
        x = self._sample_prior(n_samples, self.block_size).to(self.device)
        x_accum = torch.cat((x_accum, x), dim=1)

      # compute logits in a sliding window (context passed to model can't exceed context_size)
      end_idx = (stride_num + 1) * self.block_size
      start_idx = max(end_idx - context_size, 0)
      fwd_idx = torch.arange(start_idx, end_idx)

      dt = 1 / num_steps
      p_x0_cache = None
      timesteps = torch.linspace(1, 0, num_steps, device=self.device)
      t = 1
      for i in range(num_steps):
        if self.mask_index not in x_accum:
          break

        # faster (equivalent) sampler from zheng et al (2025)
        if self.config.sampling.first_hitting:
          u = np.random.rand()
          num_masked = (x_accum[:, fwd_idx] == self.mask_index).sum(-1).item()
          t *= u**(1 / num_masked)
        else:
          t = timesteps[i]

        p_x0_cache, x_next = self._ddpm_caching_update(
            x=x_accum[:, fwd_idx],
            t=t * ones,
            dt=dt,
            p_x0=p_x0_cache,)
        if p_x0_cache is None:
          sampling_steps += 1
       
        x_accum[:, fwd_idx] = x_next

      # [Modify] Update KV cache explicitly after the block is fully generated
      # if self.config.sampling.kv_cache:
      #     # Extract the just-generated block (last block_size tokens)
      #     # x_next here contains [context, new_block], we only need new_block for the forward pass call that updates cache
      #     current_block = x_accum[:, fwd_idx][:, -self.block_size:]
      #     _ = self.forward(current_block, t * ones, sample_mode=True, store_kv=True)
      
      # check if we need to resample (or stop sampling for variable-length sampling)
      if x_accum.shape[1] > 256:
        stop, x_accum = self._check_stop_conds(x_accum)
        if (stop and not self.config.sampling.var_length) \
          or (stop and x.shape[-1] == 1):
          return None, None
        elif stop:
          break
        
    # [Modify] Truncate to the exact requested length (handling the padded last block)
    # if seqlen is not None:
    #     x_accum = x_accum[:, :seqlen]
    
    return x_accum, sampling_steps
  
  def _compute_entropy(self, x):
    _, counts = torch.unique(x, return_counts=True, sorted=False)
    entropy = torch.special.entr(counts.float() / counts.sum()).sum()
    return entropy
  
  def _check_stop_conds(self, x):
    """Check if sampling should stop based on 1) eos, 2) entropy, or 3) likelihood.
    Entropy/likelihood evaluated on last 256 token-block.
    
    Args:
      x: torch.Tensor, current sample.
    Returns:
      stop: bool, whether to stop sampling.
      x: torch.Tensor, sample (potentially truncated for variable-length sampling).
    """
    stop = False # stop sampling?
    truncate_idx = None # truncate sample? (variable-length sampling only)

    # CRITERION 2: always stop sampling if entropy is low
    entropy = self._compute_entropy(x[:, -256:])
    if entropy < 4:
      stop = True

    # for variable length sampling, check if we should stop
    # sampling, and where to truncate the sample
    if self.config.sampling.var_length:
      # CRITERION 1: stop at sampled EOS token
      if len(torch.where(x == self.tokenizer.eos_token_id)[0]) > 1:
        stop = True
        eos_idx = torch.where(x == self.tokenizer.eos_token_id)
        if len(eos_idx[0]) > 1:
          truncate_idx = min(eos_idx[1][1]+1, x.shape[1])

      # CRITERION 2: stop if entropy/likelihood is low
      if entropy < 4:
        stop = True
        truncate_idx = x.shape[1] - 256

    # truncate sample (variable-length sampling only)
    if truncate_idx is not None:
      x = x[:, :truncate_idx]
      if x.ndim == 1:
        x = x.unsqueeze(0)

    return stop, x
