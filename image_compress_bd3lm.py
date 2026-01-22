import os
import torch
import torch.nn.functional as F
import hydra
import lightning as L
import numpy as np
import time
from tqdm import tqdm
from omegaconf import OmegaConf

import diffusion_inf
from code_utils import arithmetic_coder
from code_utils.ac_utils import normalize_pdf_for_arithmetic_coding
from code_utils.pixel_token_dict import compute_pixel_token_ids, tokenid_to_pixel
import utils
import data_loaders 

# --- Hydra Resolvers ---
OmegaConf.register_new_resolver('cwd', os.getcwd)
OmegaConf.register_new_resolver('device_count', torch.cuda.device_count)
OmegaConf.register_new_resolver('eval', eval)
OmegaConf.register_new_resolver('div_up', lambda x, y: (x + y - 1) // y)
OmegaConf.register_new_resolver("sqrt", lambda x: int(x**0.5))
OmegaConf.register_new_resolver("compute_num_patches", 
                                lambda num_images, image_size, patch_size: 
                                    int(num_images * (image_size / patch_size)**2 * 3))
OmegaConf.register_new_resolver("calc", lambda expr: eval(expr))

def get_confidence_simple(log_probs_pixel, mask_indices):
    """
    基于最大概率计算置信度
    """
    # log_probs_pixel: [1, seq_len, 256]
    # 转 double 防止溢出，计算 softmax
    probs = torch.softmax(log_probs_pixel, dim=-1)
    max_probs, _ = torch.max(probs, dim=-1) 
    confidences = max_probs[0, mask_indices]
    return confidences

def get_confidence_entropy(logits, mask_indices):
    """
    基于负熵计算置信度
    """
    log_probs = F.log_softmax(logits, dim=-1)
    probs = torch.exp(log_probs)
    
    # 计算熵: H(x) = - sum(p * log(p))
    entropy = -torch.sum(probs * log_probs, dim=-1) # [batch, seq_len]
    selected_entropy = entropy[0, mask_indices]
    confidences = -selected_entropy
    return confidences


def compress_single_patch(model, tokenizer, config, current_block_ids, pixel_token_ids_tensor, device, encoder):
    """
    对单个 Patch (Block) 执行扩散去噪编码
    """
    block_size = len(current_block_ids)
    num_steps = config.algo.T  # 扩散步数
    
    # 1. 构造初始状态
    # current_block_masked: [1, block_size]，初始全为 Mask
    current_block_masked = torch.full((1, block_size), model.mask_index, device=device, dtype=torch.long)
    true_block_tensor = torch.tensor([current_block_ids], device=device)
    maskable_mask = torch.ones_like(current_block_masked, dtype=torch.bool)
    
    # 时间步 (Time Schedule)
    timesteps = torch.linspace(1, 0, num_steps, device=device)
    t = 1
    # 2. 扩散逆过程循环
    for i in range(num_steps):
        current_mask_indices = torch.nonzero(maskable_mask[0]).flatten()
        num_current_masks = len(current_mask_indices)
        if num_current_masks == 0:
            break
        
        t = timesteps[i]  # 当前时间步
        # 计算噪声水平 Sigma
        _, move_chance_t = model.noise(t)
        sigma = model._sigma_from_p(move_chance_t).to(device)
        sigma = sigma.view(1, 1).repeat(1, 1)  # [Batch=1, 1]

        with torch.no_grad():
            # --- Forward (NAR within block) ---
            model_input = current_block_masked
            raw_logits = model.forward(model_input, sigma, sample_mode=True, store_kv=False)
        
        # 3. Logits 约束与置信度计算
        logits_pixel = raw_logits[:, :, pixel_token_ids_tensor] 
        confidences = get_confidence_entropy(logits_pixel, current_mask_indices)

        # 4. 确定性策略 (Top-K Schedule)
        sorted_indices = torch.argsort(confidences, descending=True)
        remaining_steps = num_steps - 1 - i
        ratio = 1.0 / (remaining_steps + 1)
        k = int(num_current_masks * ratio)
        k = max(1, min(k, num_current_masks))
        
        # 获取本轮要压缩的 mask 在 x_input_window 中的绝对索引
        target_indices = current_mask_indices[sorted_indices[:k]]

        # 5. 逐 Token 编码与更新
        for idx in target_indices:
            idx = idx.item()
            true_token_id = true_block_tensor[0, idx].item()
            true_pixel_value = tokenid_to_pixel(true_token_id, tokenizer)
            
            # 获取概率分布 (Double 精度)
            logit_vec = logits_pixel[0, idx].double()
            prob_dist = torch.softmax(logit_vec, dim=-1).cpu().numpy()
            
            # 算术编码
            encoder.encode(normalize_pdf_for_arithmetic_coding(prob_dist), true_pixel_value)
            
            # 填入真实 Token (去噪)
            current_block_masked[0, idx] = true_token_id
            maskable_mask[0, idx] = False
            
    return current_block_masked

def run_compression_pipeline(model, tokenizer, config, data_iterator, device):
    """
    执行完整的图像数据集压缩流程：迭代、状态管理、I/O
    """
    # 1. 初始化设置
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'image_io')
    os.makedirs(output_path, exist_ok=True)
    
    # 获取 Reset 策略参数
    reset_channel_context = getattr(config.sampling, 'reset_channel_context', True)
    output_file = os.path.join(output_path, f'compressed_output_bd3lm.txt')
    with open(output_file, "w") as f: pass

    # 准备 Token 映射
    pixel_token_ids = compute_pixel_token_ids(tokenizer)
    pixel_token_ids_tensor = torch.tensor(pixel_token_ids, device=device)

    # 计算通道切换阈值
    H_patch = config.data.patch_size
    IMG_H = config.data.image_size_test
    patches_per_channel = np.square(IMG_H // H_patch)
    
    # 强制初始化 KV Cache
    config.sampling.kv_cache = True
    model.backbone.reset_kv_cache()
    
    # 状态变量
    prev_frame_id = -1
    patch_counter_in_frame = 0 
    total_bits, total_pixels = 0, 0
    
    print(f"开始压缩流程: Image {IMG_H}x{IMG_H}, Patch {H_patch}x{H_patch}")
    
    # 2. 数据集迭代
    pbar = tqdm(data_iterator)
    for data, frame_id in pbar:
        # KV Cache 重置逻辑
        need_reset = False
        # 1. 新图片
        if frame_id != prev_frame_id:
            print(f"--- New pic (frame_id: {frame_id}), KV Cache reset ---")
            need_reset = True
            prev_frame_id = frame_id
            patch_counter_in_frame = 0
        # 2. 新通道
        elif patch_counter_in_frame > 0 and (patch_counter_in_frame % patches_per_channel == 0):
            print(f"--- Channel Switch, KV Cache reset: {reset_channel_context} ---")
            if reset_channel_context:
                need_reset = True
        if need_reset:
            model.backbone.reset_kv_cache()
        
        # 数据准备
        # data [H, W, C] -> flatten -> tokens
        seq_array = data.reshape(1, model.block_size)
        flattened_array = seq_array.flatten()
        num_str_tokens = [str(num) for num in flattened_array]
        current_block_ids = tokenizer.convert_tokens_to_ids(num_str_tokens)
        
        # 压缩核心逻辑
        # 每个 Patch 独立编码为一行 Bitstream
        output_bits = []
        encoder = arithmetic_coder.Encoder(
            base=config.data.ac_coder_base,
            precision=config.data.ac_coder_precision,
            output_fn=output_bits.append,
        )

        # Block 级压缩
        filled_block_tensor = compress_single_patch(
            model, tokenizer, config, current_block_ids, 
            pixel_token_ids_tensor, device, encoder
        )
        
        encoder.terminate()
        compressed_bits = "".join(map(str, output_bits))
        
        # 更新 KV Cache (Sliding Window)
        with torch.no_grad():
            sigma_zero = torch.zeros((1, 1), device=device)
            _ = model.forward(filled_block_tensor, sigma_zero, sample_mode=True, store_kv=True)
            
        patch_counter_in_frame += 1

        # 结果写入
        bits_to_write = compressed_bits + '1'  # Stop Bit 保护
        num_bits = len(bits_to_write)
        total_bits += num_bits
        total_pixels += model.block_size
        
        with open(output_file, "a") as f:
            f.write(bits_to_write + '\n')
            
        pbar.set_description(f"BPP: {total_bits/total_pixels:.4f}")

    print("\n" + "="*30)
    print(f"原始像素Token数: {total_pixels}")
    print(f"压缩后比特数: {total_bits}")
    print(f"总平均 BPP: {total_bits/total_pixels:.4f}")
    print(f"结果已保存至: {output_file}")
    print("="*30)

@hydra.main(version_base=None, config_path='configs', config_name='config_inf')
def main(config):
    L.seed_everything(config.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = utils.get_tokenizer(config)
    
    # 1. 加载模型
    print(f"Loading model from {config.sampling.checkpoint_path}...")
    model = diffusion_inf.Diffusion(config=config, tokenizer=tokenizer)
    model.to(device)
    model.backbone.eval()
    model.noise.eval()
    
    # 开启矩阵乘法精度优化
    # torch.set_float32_matmul_precision('high')
    
    # 2. 加载数据
    if config.data.test_dataset == "CIFAR10":
        test_dataset_path = os.path.join(config.data.dataset_root, "CIFAR10", "cifar10_test_one")
    elif config.data.test_dataset  == "DIV2K":
        test_dataset_path  = os.path.join(config.data.dataset_root, "DIV2K", "DIV2K_LR_test")
    elif config.data.test_dataset == "ImageNet":
        test_dataset_path  = os.path.join(config.data.dataset_root, "ImageNet", "test_unified")
    print(f"Dataset Path: {test_dataset_path}")
        
    data_iterator = data_loaders.get_image_iterator(
                    patch_size=config.data.patch_size,
                    num_chunks=config.data.num_patches_test,
                    is_channel_wised=config.data.is_channel_wised,
                    is_seq=False, 
                    data_path=test_dataset_path)

    # 3. 执行压缩任务
    run_compression_pipeline(model, tokenizer, config, data_iterator, device)

if __name__ == '__main__':
    main()