import os
import fsspec
import hydra
import lightning as L
import omegaconf
import rich.syntax
import rich.tree
import torch

import diffusion_train
import utils
from data_loaders import Div2kPatchDataset

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
    data_path=os.path.join(config.loader.data_dir, "train"),
    tokenizer=tokenizer,
    samples_per_image=100,
    is_channel_wised=config.data.is_channel_wised,
    shuffle=True,
    split='train'
  )

  valid_set = Div2kPatchDataset(
    data_path=os.path.join(config.loader.data_dir, "valid"),
    tokenizer=tokenizer,
    samples_per_image=1,
    is_channel_wised=config.data.is_channel_wised,
    shuffle=False,
    split='valid'
  )
  
  train_loader = torch.utils.data.DataLoader(
    train_set,
    batch_size=config.loader.batch_size,
    num_workers=config.loader.num_workers,
    pin_memory=config.loader.pin_memory,
    shuffle=False,  # 必须为 False，乱序已在 Dataset 内部通过 shuffle=True 处理
    persistent_workers=True if config.loader.num_workers > 0 else False,
    drop_last=True
  )

  valid_loader = torch.utils.data.DataLoader(
    valid_set,
    batch_size=config.loader.eval_batch_size,
    num_workers=config.loader.num_workers,
    pin_memory=config.loader.pin_memory,
    shuffle=False, 
    persistent_workers=True if config.loader.num_workers > 0 else False,
    drop_last=True
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