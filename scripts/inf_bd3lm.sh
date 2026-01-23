python compress_bd3lm_optim.py \
    algo.backbone=hf_dit \
    algo.T=3 \
    sampling.first_hitting=False \
    sampling.reset_channel_context=True \
    sampling.batch_size=10 \
    model.length=1024 \
    block_size=16 \
    data.test_dataset=CIFAR10 \
    data.image_size_test=32 \
    data.num_images_test=10 \
    model.attn_backend=sdpa \