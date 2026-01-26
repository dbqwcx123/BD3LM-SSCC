block_size=64
python compress_bd3lm_optim.py \
    algo.backbone=dit \
    algo.T=5 \
    block_size=${block_size} \
    loader.eval_batch_size=10 \
    sampling.checkpoint_path="/mnt/e/1Master1/0Science_Research/0DLM-SSCC/Model/bd3lm/bd3lm-owt-block_size${block_size}/best.ckpt" \
    sampling.first_hitting=False \
    sampling.reset_channel_context=True \
    model.length=1024 \
    data.test_dataset=CIFAR10 \
    data.image_size_test=32 \
    data.num_images_test=10 \
    model.attn_backend=sdpa \