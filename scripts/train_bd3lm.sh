python train_bd3lm.py \
    loader.data_dir=/mnt/e/1Master1/0Science_Research/0DLM-SSCC/Dataset/DIV2K/DIV2K_LR_unified/X4\
    loader.global_batch_size=64 \
    loader.eval_global_batch_size=64 \
    loader.batch_size=2 \
    loader.eval_batch_size=2 \
    model=small \
    algo=bd3lm \
    algo.backbone=hf_dit \
    algo.clip_search_widths=[0.5,0.6,0.7,0.8,0.9] \
    model.length=1024 \
    block_size=64 \
    wandb.name=bd3lm-div2k-block_size64-260126 \
    mode=train \
    model.attn_backend=flex \
    training.resample=True \
    training.from_pretrained=/mnt/e/1Master1/0Science_Research/0DLM-SSCC/Model/bd3lm/bd3lm-owt-block_size1024-pretrain \