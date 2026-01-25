python train_bd3lm.py \
    loader.data_dir=/mnt/e/1Master1/0Science_Research/0DLM-SSCC/Dataset/DIV2K/DIV2K_HR_p128\
    loader.global_batch_size=128 \
    loader.eval_global_batch_size=128 \
    loader.batch_size=4 \
    loader.eval_batch_size=4 \
    model=small \
    algo=bd3lm \
    algo.backbone=hf_dit \
    algo.clip_search_widths=[0.5,0.6,0.7,0.8,0.9] \
    model.length=1024 \
    block_size=64 \
    wandb.name=bd3lm-owt-block_size64 \
    mode=train \
    model.attn_backend=flex \
    training.resample=True \
    training.from_pretrained=/mnt/e/1Master1/0Science_Research/0DLM-SSCC/Model/bd3lm/bd3lm-owt-block_size1024-pretrain/model.safetensors \
    eval.checkpoint_path=/mnt/e/1Master1/0Science_Research/0DLM-SSCC/Model/bd3lm/bd3lm-owt-block_size1024-pretrain \