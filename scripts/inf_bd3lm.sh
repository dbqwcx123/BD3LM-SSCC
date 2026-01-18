python inf_bd3lm.py \
    algo.backbone=hf_dit \
    algo.T=20 \
    model.length=20 \
    block_size=16 \
    eval.checkpoint_path=/mnt/e/1Master1/0Science_Research/0DLM-SSCC/Model/bd3lm/bd3lm-owt-block_size16 \
    model.attn_backend=sdpa \
    training.ema=0