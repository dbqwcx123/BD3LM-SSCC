import os
import fsspec
import hydra
import lightning as L
import omegaconf
import rich.syntax
import rich.tree
import torch
import transformers

import dataloader
import diffusion
import utils

omegaconf.OmegaConf.register_new_resolver(
  'cwd', os.getcwd)
omegaconf.OmegaConf.register_new_resolver(
  'device_count', torch.cuda.device_count)
omegaconf.OmegaConf.register_new_resolver(
  'eval', eval)
omegaconf.OmegaConf.register_new_resolver(
  'div_up', lambda x, y: (x + y - 1) // y)


def _load_from_checkpoint(config, tokenizer):
  if 'hf' in config.algo.backbone:
    return diffusion.Diffusion(
      config, tokenizer=tokenizer).to('cuda')
  
  return diffusion.Diffusion.load_from_checkpoint(
    config.eval.checkpoint_path,
    tokenizer=tokenizer,
    config=config,
    strict=False,
    weights_only=False).to('cuda')

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

def generate_samples(config, logger, tokenizer):
  logger.info('Generating samples.')
  model = _load_from_checkpoint(config=config,  tokenizer=tokenizer)
  if config.eval.disable_ema:
    logger.info('Disabling EMA.')
    model.ema = None
  text_samples = model.restore_model_and_sample(num_steps=config.algo.T)
  print('Text samples:', text_samples)
  return text_samples


@hydra.main(version_base=None, config_path='configs',
            config_name='config_inf')
def main(config):
  """Main entry point for training."""
  L.seed_everything(config.seed)
  _print_config(config, resolve=True, save_cfg=True)
  
  logger = utils.get_logger(__name__)
  tokenizer = dataloader.get_tokenizer(config)

  samples = generate_samples(config, logger, tokenizer)


if __name__ == '__main__':
  main()