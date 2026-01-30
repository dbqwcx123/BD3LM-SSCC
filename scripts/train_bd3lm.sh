python train_bd3lm.py \
    loader.data_dir=/data_133/rtq/Dataset/DIV2K/DIV2K_LR_unified/X4\
    loader.global_batch_size=256 \
    loader.eval_global_batch_size=256 \
    loader.batch_size=4 \
    loader.eval_batch_size=4 \
    model=small \
    algo=bd3lm \
    algo.backbone=hf_dit \
    algo.clip_search_widths=[0.5,0.6,0.7,0.8,0.9] \
    model.length=1024 \
    block_size=256 \
    wandb.name=bd3lm-div2k-block_size256-260130 \
    mode=train \
    model.attn_backend=flex \
    training.resample=True \
    training.from_pretrained=/data_133/rtq/Model/bd3lm/bd3lm-owt-block_size1024-pretrain \
    data.tokenizer_name_or_path=/data_133/rtq/Model/gpt2 \