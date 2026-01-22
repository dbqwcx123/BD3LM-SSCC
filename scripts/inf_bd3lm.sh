python image_compress_bd3lm.py \
    algo.backbone=hf_dit \
    algo.T=3 \
    sampling.first_hitting=False \
    sampling.reset_channel_context=True \
    model.length=1024 \
    block_size=16 \
    data.test_dataset=CIFAR10 \
    data.num_images_test=1 \
    model.attn_backend=sdpa \