import functools
import itertools
import json
import math
import os
import re
import shutil
import typing
import urllib
import zipfile
from typing import Optional

import datasets
import fsspec
import requests
import tokenizers
import torch
import transformers

import utils

LOGGER = utils.get_logger(__name__)


def _group_texts(examples, block_size, bos, eos, insert_special_tokens=True):
  # Concatenate all texts.
  concatenated_examples = list(itertools.chain(* examples['input_ids']))
  total_length = len(concatenated_examples)
  # TODO(yair): look into not dropping the remainder but rather padding it.
  # We drop the small remainder, and if the total_length < block_size - 2
  # we exclude this batch and return an empty dict.
  # We could add padding if the model supported it instead of
  # this drop, you can customize this part to your needs.
  if insert_special_tokens:
    new_block_size = block_size - 2  # [BOS] and [EOS] to be added
  else:
    new_block_size = block_size
  total_length = (total_length // new_block_size) * new_block_size
  # Split by chunks of max_len.
  result = {}
  _values = []
  _attn_masks = []
  for i in range(0, total_length, new_block_size):
    if insert_special_tokens:
      _values.append(
        [bos]
        + concatenated_examples[i : i + new_block_size]
        + [eos])
    else:
      _values.append(
        concatenated_examples[i : i + new_block_size]
      )
    _attn_masks.append(torch.ones(block_size))
  result['input_ids'] = _values
  result['attention_mask'] = _attn_masks
  return result


def get_dataset(
    dataset_name, tokenizer, mode, cache_dir,
    block_size=1024, num_proc=len(os.sched_getaffinity(0)),
    insert_eos=True, insert_special_tokens=True):
  eos_tag = ''
  if not insert_eos:
    eos_tag = '_eosFalse'
  if not insert_special_tokens:
    eos_tag = '_specialFalse'
  filename = f'{dataset_name}_{mode}_bs{block_size}_wrapped{eos_tag}.dat'
  _path = os.path.join(cache_dir, filename)
  
  if utils.fsspec_exists(_path):
    LOGGER.info(f'Loading data from: {_path}')
    return datasets.load_from_disk(_path).with_format('torch')
  LOGGER.info(f'Generating new data at: {_path}')
  
  if dataset_name == 'openwebtext-train':
    dataset = datasets.load_dataset(
      'openwebtext',
      split='train[:-100000]',
      cache_dir=cache_dir,
      trust_remote_code=True)
  elif dataset_name == 'openwebtext-valid':
    dataset = datasets.load_dataset(
      'openwebtext',
      split='train[-100000:]',
      cache_dir=cache_dir,
      trust_remote_code=True)
  else:
    dataset = datasets.load_dataset(
      dataset_name,
      cache_dir=cache_dir,
      trust_remote_code=True)

  if dataset_name in ['openwebtext-train', 'openwebtext-valid']:
    data = dataset
  else:
    data = dataset[mode]

  EOS = tokenizer.encode(tokenizer.eos_token)[0]
  BOS = tokenizer.encode(tokenizer.bos_token)[0]

  def preprocess_and_tokenize(example):
    text = example['text']
    
    tokenizer.padding_side = 'right'
    tokenizer.truncation_side = 'right'

    tokens = tokenizer(text,
                        add_special_tokens=False,
                        return_attention_mask=False,
                        return_token_type_ids=False)
    if insert_eos:
      tokens = {'input_ids':
                [t + [EOS] for t in tokens['input_ids']]}
    # Still missing BOS, but will be added in group_texts
    return tokens

  tokenized_dataset = data.map(
    preprocess_and_tokenize,
    batched=True,
    num_proc=num_proc,
    load_from_cache_file=True,
    desc='Tokenizing')
  tokenized_dataset = tokenized_dataset.remove_columns('text')

  group_texts = functools.partial(
    _group_texts, block_size=block_size, bos=BOS, eos=EOS, insert_special_tokens=insert_special_tokens)
  
  chunked_dataset = tokenized_dataset.map(
    group_texts,
    batched=True,
    num_proc=num_proc,
    load_from_cache_file=True,
    desc='Grouping')
  chunked_dataset.save_to_disk(_path)
  chunked_dataset = chunked_dataset.with_format('torch')
  return chunked_dataset


def get_tokenizer(config):
  tokenizer = transformers.AutoTokenizer.from_pretrained(
    config.data.tokenizer_name_or_path)

  if (isinstance(tokenizer, transformers.GPT2TokenizerFast)
      or isinstance(tokenizer, transformers.GPT2Tokenizer)):
    tokenizer._tokenizer.post_processor = tokenizers.processors.BertProcessing(
      (tokenizer.bos_token, tokenizer.bos_token_id),
      (tokenizer.eos_token, tokenizer.eos_token_id))

  # For wrapped batches:
  #  [BOS] sent1 [EOS] sent2-fragment [EOS]
  #  [BOS] sent2-fragment [EOS] sent3 [EOS]
  if tokenizer.bos_token is None:
    if tokenizer.cls_token is None:
      raise AttributeError(
        'Tokenizer must have a bos_token or '
        f'cls_token: {tokenizer}')
    tokenizer.bos_token = tokenizer.cls_token
  if tokenizer.eos_token is None:
    if tokenizer.sep_token is None:
      raise AttributeError(
        'Tokenizer must have a eos_token '
        f'or sep_token: {tokenizer}')
    tokenizer.eos_token = tokenizer.sep_token
  if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

  return tokenizer
    

def get_dataloaders(config, tokenizer):
  num_gpus = torch.cuda.device_count()
  if config.trainer.accumulate_grad_batches > 1:
    assert (config.loader.global_batch_size
            == (config.loader.batch_size
                * config.trainer.num_nodes
                * num_gpus
                * config.trainer.accumulate_grad_batches))
    if config.loader.global_batch_size % (
      num_gpus * config.trainer.accumulate_grad_batches) != 0:
      raise ValueError(
        f'Train Batch Size {config.training.batch_size}'
        f'not divisible by {num_gpus} gpus with accumulation '
        f'{config.trainer.accumulate_grad_batches}.')
  if config.loader.global_batch_size % (
    num_gpus * config.trainer.accumulate_grad_batches) != 0:
    raise ValueError(
      f'Train Batch Size {config.training.batch_size}'
      f'not divisible by {num_gpus} gpus with accumulation '
      f'{config.trainer.accumulate_grad_batches}.')
  if config.loader.eval_global_batch_size % num_gpus != 0:
    raise ValueError(
      f'Eval Batch Size for {config.eval.batch_size} '
      f'not divisible by {num_gpus}.')
  
  train_set = get_dataset(
    config.data.train,
    tokenizer,
    mode='train',
    insert_eos=True,
    insert_special_tokens=False,
    cache_dir=config.data.cache_dir,
    block_size=config.model.length)
  
  valid_set = get_dataset(
    config.data.valid,
    tokenizer,
    insert_eos=False,
    insert_special_tokens=False,
    mode='validation',
    cache_dir=config.data.cache_dir,
    block_size=config.model.length,)

  train_loader = torch.utils.data.DataLoader(
    train_set,
    batch_size=config.loader.batch_size,
    num_workers=config.loader.num_workers,
    pin_memory=config.loader.pin_memory,
    shuffle=True,
    persistent_workers=True)
  train_loader.tokenizer = tokenizer
  
  valid_loader = torch.utils.data.DataLoader(
    valid_set,
    batch_size=config.loader.eval_batch_size,
    num_workers=config.loader.num_workers,
    pin_memory=config.loader.pin_memory,
    shuffle=False)
  # Will be used in generative perplexity calculation
  valid_loader.tokenizer = tokenizer

  return train_loader, valid_loader


# Samplers adapted from: https://github.com/Dao-AILab/flash-attention/blob/main/training/src/datamodules/fault_tolerant_sampler.py


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
