import os
import torch
import hydra
import lightning as L
import logging
import numpy as np
import time
from tqdm import tqdm
from omegaconf import OmegaConf

import diffusion_inf
from code_utils import arithmetic_coder
from code_utils.ac_utils import normalize_pdf_for_arithmetic_coding
import constants, utils

# --- Hydra Resolvers ---
OmegaConf.register_new_resolver('cwd', os.getcwd)
OmegaConf.register_new_resolver('device_count', torch.cuda.device_count)
OmegaConf.register_new_resolver('eval', eval)
OmegaConf.register_new_resolver('div_up', lambda x, y: (x + y - 1) // y)

def get_confidence(log_probs, mask_indices):
    """
    基于 log_probs 计算置信度 (需要 exp)
    """
    probs = torch.exp(log_probs)
    max_probs, _ = torch.max(probs, dim=-1) 
    confidences = max_probs[0, mask_indices]
    return confidences

def compress_semi_ar(model, tokenizer, config, full_input_ids, device):
    """
    BD3LM 半自回归编码 (复刻 _semi_ar_sampler 逻辑)
    """
    # 1. 初始化设置
    seqlen = len(full_input_ids)        # 序列长度
    block_size = model.block_size       # 块大小 (如 16)
    num_steps = config.algo.T           # 扩散步数
    context_size = config.model.length  # 模型最大上下文长度 (如 512, 1024)
    
    # 将输入填充为 block_size 的整数倍
    pad_len = (block_size - seqlen % block_size) % block_size
    if pad_len > 0: full_input_ids = full_input_ids + [tokenizer.pad_token_id] * pad_len
    num_strides = seqlen // block_size
    full_input_tensor = torch.tensor([full_input_ids], device=device) # [1, total_len]

    # 初始化算术编码器
    output_bits = []
    encoder = arithmetic_coder.Encoder(
        base=constants.ARITHMETIC_CODER_BASE,
        precision=constants.ARITHMETIC_CODER_PRECISION,
        output_fn=output_bits.append,
    )

    # 重置 KV Cache
    if config.sampling.kv_cache:
        model.backbone.reset_kv_cache(eval_batch_size=1)
    
    # x_accum 用于存储已处理的文本 (Context)，初始为空或 BOS，x_accum 动态增长
    x_accum = torch.tensor([], device=device, dtype=torch.long).reshape(1, 0)

    print(f"开始压缩: 总长度 {seqlen}, 块大小 {block_size}, 总块数 {num_strides}")

    pbar = tqdm(range(num_strides))
    for stride_num in pbar:
        pbar.set_description(f"Bits: {len(output_bits)}")
        
        # 当前 Block 的真实数据
        block_start = stride_num * block_size
        block_end = (stride_num + 1) * block_size
        true_block = full_input_tensor[:, block_start:block_end] # [1, block_size]
        # 构造初始状态：当前 Block 全为 Mask
        current_block_masked = torch.full_like(true_block, model.mask_index)
        # 将 Masked Block 拼接到历史 Context 后
        x_accum = torch.cat((x_accum, current_block_masked), dim=1)
        
        # 滑动窗口
        end_idx = (stride_num + 1) * block_size
        start_idx = max(end_idx - context_size, 0)
        fwd_idx = torch.arange(start_idx, end_idx, device=device) # 当前输入模型的索引范围
        
        # 截取 x_accum，当前 Block 在 fwd_idx 中的相对位置
        x_input_window = x_accum[:, fwd_idx]

        # --- Block 内扩散压缩循环 ---
        maskable_mask = torch.zeros_like(x_input_window, dtype=torch.bool)
        maskable_mask[:, -block_size:] = True 
        
        # 时间步 (Time Schedule)
        timesteps = torch.linspace(1, 0, num_steps, device=device)
        t = 1
        for i in range(num_steps):
            current_mask_indices = torch.nonzero(maskable_mask[0]).flatten()
            num_current_masks = len(current_mask_indices)
            if num_current_masks == 0:
                break
                
            t = timesteps[i]  # 当前时间步
            # 计算噪声水平 Sigma
            _, move_chance_t = model.noise(t)
            sigma = model._sigma_from_p(move_chance_t).to(device)
            sigma = sigma.view(1, 1).repeat(x_input_window.shape[0], 1)  # 扩展维度以匹配输入 [Batch, 1]

            with torch.no_grad():
                # --- Forward (带 KV Cache 优化) ---
                # 在 Block 内部迭代时不更新 KV Cache，只利用之前的 Cache
                if config.sampling.kv_cache:
                    model_input = x_input_window[:, -block_size:]
                else:
                    model_input = x_input_window
                
                log_probs = model.forward(model_input, sigma, sample_mode=True, store_kv=False)
            
            # --- 确定性压缩策略 (Confidence-based) ---
            if config.sampling.kv_cache:
                # 当前 Block 内的 mask 索引 -> 相对于 log_probs 的索引
                # current_mask_indices 是相对于 x_input_window 的全局索引
                window_len = x_input_window.shape[1]
                offset = window_len - block_size
                rel_mask_indices = current_mask_indices - offset
                valid_mask = rel_mask_indices >= 0
                rel_mask_indices = rel_mask_indices[valid_mask]
                
                confidences = get_confidence(log_probs, rel_mask_indices)
            else:
                confidences = get_confidence(log_probs, current_mask_indices)

            # 排序与选择 Top-K
            sorted_indices = torch.argsort(confidences, descending=True)
            # Schedule: 1 / (remaining_steps + 1)
            remaining_steps = num_steps - 1 - i
            ratio = 1.0 / (remaining_steps + 1)
            k = int(num_current_masks * ratio)
            k = max(1, min(k, num_current_masks))
            
            # 获取本轮要压缩的 mask 在 x_input_window 中的绝对索引
            target_indices_global = current_mask_indices[sorted_indices[:k]]

            # --- 编码与状态更新 ---
            for global_idx in target_indices_global:
                global_idx = global_idx.item()
                
                # 获取真实 Token (注意 x_accum 和 full_input_tensor 的对齐)
                real_abs_pos = start_idx + global_idx
                true_token = full_input_tensor[0, real_abs_pos].item()
                
                # 获取概率分布
                if config.sampling.kv_cache:
                    rel_idx = global_idx - (x_input_window.shape[1] - block_size)
                    prob_dist = torch.exp(log_probs[0, rel_idx]).cpu().numpy()
                else:
                    prob_dist = torch.exp(log_probs[0, global_idx]).cpu().numpy()
                
                encoder.encode(normalize_pdf_for_arithmetic_coding(prob_dist), true_token)
                
                # 更新状态
                x_input_window[0, global_idx] = true_token
                maskable_mask[0, global_idx] = False
                x_accum[0, start_idx + global_idx] = true_token

        # --- 存储 KV ---
        if config.sampling.kv_cache:
            with torch.no_grad():
                # 使用 sigma = 0 进行纯推理更新，sigma 此时仅用于 embedding 
                batch_size = x_accum.shape[0]
                sigma_zero = torch.zeros((batch_size,1), device=device) 
                model_input = x_accum[:, -block_size:]  # 只输入刚处理完的 Block
                _ = model.forward(model_input, sigma_zero, sample_mode=True, store_kv=True)

    encoder.terminate()
    return "".join(map(str, output_bits)), seqlen

@hydra.main(version_base=None, config_path='configs', config_name='config_inf')
def main(config):
    L.seed_everything(config.seed)
    logger = utils.get_logger(__name__)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = utils.get_tokenizer(config)
    
    # 加载模型 (自动处理 backbone 和 sdpa/flex attention)
    print(f"Loading model from {config.eval.checkpoint_path}...")
    model = diffusion_inf.Diffusion(config=config, tokenizer=tokenizer)
    model.to(device)
    model.backbone.eval()
    model.noise.eval()

    # 路径设置
    current_script_path = os.path.abspath(__file__)
    project_root = os.path.dirname(current_script_path)
    input_file = os.path.join(project_root, 'text_io', 'demo_origin.txt')
    
    if not os.path.exists(input_file):
        print(f"错误: 找不到输入文件 {input_file}")
        return
    
    # 读取输入文件 (文本形式)
    with open(input_file, "r") as f:
        text = f.read().strip()
        text = text * 8
    
    input_ids = tokenizer.encode(text, add_special_tokens=True, truncation=False)
    
    # 压缩
    start_time = time.time()
    compressed_bits, orig_len = compress_semi_ar(model, tokenizer, config, input_ids, device)
    end_time = time.time()
    
    print("\n" + "="*30)
    print(f"Original Tokens: {orig_len}")
    print(f"Compressed Bits: {len(compressed_bits)}")
    print(f"BPC (Bits Per Char): {len(compressed_bits)/len(text):.2f}")
    print(f"Time: {end_time - start_time:.2f}s")
    print("="*30)

    output_file = os.path.join(project_root, 'text_io', 'compressed_output.txt')
    with open(output_file, "w") as f:
        f.write(compressed_bits)
    print(f"压缩结果已保存至: {output_file}")
    

if __name__ == '__main__':
    main()