python main.py \
    loader.eval_batch_size=1 \
    model=small \
    algo=bd3lm \
    algo.backbone=hf_dit \
    algo.T=200 \
    data=openwebtext-split \
    model.length=1000 \
    block_size=16 \
    wandb=null \
    mode=sample_eval \
    eval.checkpoint_path='../Model/bd3lm/bd3lm-owt-block_size16' \
    model.attn_backend=sdpa \
    seed=42 \
    sampling.num_sample_batches=1 \
    sampling.nucleus_p=0.9 \
    sampling.kv_cache=true \
    sampling.logdir=./sample_logs