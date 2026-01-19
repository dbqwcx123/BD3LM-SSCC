import os
import torch
import hydra
import logging
import numpy as np
import time
from tqdm import tqdm
from omegaconf import OmegaConf

import diffusion_inf
from code_utils import arithmetic_coder
from code_utils.ac_utils import normalize_pdf_for_arithmetic_coding

# --- Hydra Resolvers (与 inf_bd3lm.py 保持一致) ---
try:
    OmegaConf.register_new_resolver('cwd', os.getcwd)
    OmegaConf.register_new_resolver('device_count', torch.cuda.device_count)
    OmegaConf.register_new_resolver('eval', eval)
    OmegaConf.register_new_resolver('div_up', lambda x, y: (x + y - 1) // y)
except:
    pass

def get_tokenizer(config):
    import transformers
    import tokenizers
    tokenizer = transformers.AutoTokenizer.from_pretrained(config.data.tokenizer_name_or_path)
    
    if tokenizer.bos_token is None and tokenizer.cls_token is not None:
        tokenizer.bos_token = tokenizer.cls_token
    if tokenizer.eos_token is None and tokenizer.sep_token is not None:
        tokenizer.eos_token = tokenizer.sep_token
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    return tokenizer

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
    复刻 diffusion_inf.py 中的 _semi_ar_sampler 逻辑进行压缩。
    """
    # 1. 初始化设置
    seqlen = len(full_input_ids)        # 序列长度
    block_size = model.block_size       # 块大小 (如 16)
    num_steps = config.algo.T           # 扩散步数
    context_size = config.model.length  # 模型最大上下文长度 (如 512, 1024)
    
    # 将输入填充为 block_size 的整数倍
    pad_len = (block_size - seqlen % block_size) % block_size
    if pad_len > 0:
        full_input_ids = full_input_ids + [tokenizer.pad_token_id] * pad_len
    
    num_strides = seqlen // block_size
    full_input_tensor = torch.tensor([full_input_ids], device=device) # [1, total_len]

    # 初始化算术编码器
    output_bits = []
    encoder = arithmetic_coder.Encoder(
        base=2,
        precision=32,
        output_fn=output_bits.append,
    )

    # **关键逻辑 1: 重置 KV Cache**
    if config.sampling.kv_cache:
        model.backbone.reset_kv_cache(eval_batch_size=1)
    
    # x_accum 用于存储已处理的文本 (Context)，初始为空或 BOS
    # 这里我们采用 _semi_ar_sampler 的逻辑，x_accum 动态增长
    x_accum = torch.tensor([], device=device, dtype=torch.long).reshape(1, 0)

    print(f"开始压缩: 总长度 {seqlen}, 块大小 {block_size}, 总块数 {num_strides}")

    pbar = tqdm(range(num_strides))
    for stride_num in pbar:
        pbar.set_description(f"Bits: {len(output_bits)}")
        # 2. 准备当前 Block 的真实数据
        block_start = stride_num * block_size
        block_end = (stride_num + 1) * block_size
        true_block = full_input_tensor[:, block_start:block_end] # [1, block_size]
        
        # 3. 构造初始状态：当前 Block 全为 Mask
        # 参考 _semi_ar_sampler: x = self._sample_prior(...)
        current_block_masked = torch.full_like(true_block, model.mask_index)
        
        # 将 Masked Block 拼接到历史 Context 后
        x_accum = torch.cat((x_accum, current_block_masked), dim=1)
        
        # **关键逻辑 2: 滑动窗口 (Context Window)**
        # 参考 _semi_ar_sampler: compute logits in a sliding window
        end_idx = (stride_num + 1) * block_size
        start_idx = max(end_idx - context_size, 0)
        fwd_idx = torch.arange(start_idx, end_idx, device=device) # 当前输入模型的索引范围
        
        # 当前 Block 在 fwd_idx 中的相对位置 (通常是最后 block_size 个)
        # 注意：如果 start_idx > 0，x_accum 已经被截断视觉，但 x_accum 变量本身随着 dim=1 增长
        # 为了传给 model，我们需要截取 x_accum
        x_input_window = x_accum[:, fwd_idx]

        # 4. 扩散压缩循环 (Block 内)
        # 初始化 Mask 状态
        maskable_mask = torch.zeros_like(x_input_window, dtype=torch.bool)
        # 只有最后 block_size 个位置（即当前 block）是 maskable 的
        maskable_mask[:, -block_size:] = True 
        
        # 准备时间步 (Time Schedule)
        # 参考 _semi_ar_sampler: timesteps = torch.linspace(1, 0, num_steps, ...)
        timesteps = torch.linspace(1, 0, num_steps, device=device)
        
        for i in range(num_steps):
            current_mask_indices = torch.nonzero(maskable_mask[0]).flatten()
            num_current_masks = len(current_mask_indices)
            if num_current_masks == 0:
                break
                
            t = timesteps[i] # 获取当前时间步
            
            # **关键逻辑 3: 计算 Sigma**
            # 参考 _ddpm_caching_update: sigma_t = self._sigma_from_p(move_chance_t)
            # 这里简化：直接通过 t 获取对应的 move_chance (即 1-alpha_bar 之类的概念)
            # 在 diffusion_inf.py 中: _, move_chance_t = self.noise(t)
            _, move_chance_t = model.noise(t)
            sigma = model._sigma_from_p(move_chance_t).to(device)
            # 扩展 sigma 维度以匹配输入 [Batch, 1] (虽然后续 forward 内部会处理)
            sigma = sigma.view(1, 1).repeat(x_input_window.shape[0], 1)

            with torch.no_grad():
                # **关键逻辑 4: Forward (带 KV Cache 优化)**
                # 如果 kv_cache 开启，diffusion_inf 逻辑是 forward(x[:, -block_size:])
                # 但前提是前面的 kv 已经存好了。我们在本 Block 循环开始前并没有存当前 Block 的 kv。
                # 在 Block 内部迭代时，我们不更新 KV Cache (store_kv=False)，只利用之前的 Cache。
                
                if config.sampling.kv_cache:
                    # 只输入当前 Block 部分，利用 Cache 中的 Past
                    model_input = x_input_window[:, -block_size:]
                else:
                    # 输入完整 Window
                    model_input = x_input_window
                
                # forward 返回 log_probs
                log_probs = model.forward(model_input, sigma, sample_mode=True, store_kv=False)
                
                # 注意：如果 kv_cache=True，输出的 log_probs 长度只有 block_size
                # 对应的索引偏移量需要调整
            
            # 5. 确定性压缩策略 (Confidence-based)
            # 如果 kv_cache=True，log_probs 对应 maskable_mask 的最后 block_size 部分
            if config.sampling.kv_cache:
                # 提取当前 Block 内的 mask 索引 (相对于 block 内部的 0~15)
                # current_mask_indices 是相对于 x_input_window 的全局索引
                # 我们需要将其转换为相对于 log_probs 的索引
                window_len = x_input_window.shape[1]
                offset = window_len - block_size
                
                # 过滤出属于当前 block 的 mask indices (理论上全是)
                rel_mask_indices = current_mask_indices - offset
                valid_mask = rel_mask_indices >= 0
                rel_mask_indices = rel_mask_indices[valid_mask]
                
                confidences = get_confidence(log_probs, rel_mask_indices)
            else:
                confidences = get_confidence(log_probs, current_mask_indices)

            # 排序与选择 Top-K
            sorted_indices = torch.argsort(confidences, descending=True)
            
            # Schedule: 1 / (remaining_steps + 1)
            # 注意 i 是从 0 到 num_steps-1，剩余步数是 num_steps - 1 - i
            remaining_steps = num_steps - 1 - i
            ratio = 1.0 / (remaining_steps + 1)
            k = int(num_current_masks * ratio)
            k = max(1, min(k, num_current_masks))
            
            # 获取本轮要解压的 mask 在 x_input_window 中的绝对索引
            target_indices_global = current_mask_indices[sorted_indices[:k]]

            # 6. 编码与状态更新
            for global_idx in target_indices_global:
                global_idx = global_idx.item()
                
                # 获取真实 Token (注意 x_accum 和 full_input_tensor 的对齐)
                # stride_num * block_size 是当前 Block 的起点
                # global_idx 是相对于 x_input_window 的位置
                # 真实的全局位置 = start_idx + global_idx
                real_abs_pos = start_idx + global_idx
                true_token = full_input_tensor[0, real_abs_pos].item()
                
                # 获取概率分布
                if config.sampling.kv_cache:
                    rel_idx = global_idx - (x_input_window.shape[1] - block_size)
                    prob_dist = torch.exp(log_probs[0, rel_idx]).cpu().numpy()
                else:
                    prob_dist = torch.exp(log_probs[0, global_idx]).cpu().numpy()
                
                encoder.encode(normalize_pdf_for_arithmetic_coding(prob_dist), true_token)
                
                # 更新 x_input_window (也同步更新 x_accum，因为我们需要切片引用或手动写回)
                x_input_window[0, global_idx] = true_token
                maskable_mask[0, global_idx] = False
                
                # 同步回 x_accum (用于下个 Block 的 Context)
                x_accum[0, start_idx + global_idx] = true_token

        # **关键逻辑 5: 更新 KV Cache (Store KV)**
        # 当前 Block 压缩完毕，所有 Token 已变成真实值。
        # 此时执行一次 forward，将当前 Block 的 KV 写入 Cache，供下一个 Block 使用。
        # 参考 diffusion_inf.py Line 186: if ... mask_index not in x_block ... store_kv=True
        if config.sampling.kv_cache:
            with torch.no_grad():
                # 使用最小 sigma (或 0) 进行纯推理更新，虽然 sigma 此时仅用于 embedding 
                # 通常使用 sigma_min 或 0 即可
                batch_size = x_accum.shape[0]
                sigma_zero = torch.zeros((batch_size,1), device=device) 
                model_input = x_accum[:, -block_size:] # 只输入刚处理完的 Block
                model.forward(model_input, sigma_zero, sample_mode=True, store_kv=True)

    encoder.terminate()
    return "".join(map(str, output_bits)), seqlen

@hydra.main(version_base=None, config_path='configs', config_name='config_inf')
def main(config):
    torch.manual_seed(config.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = get_tokenizer(config)
    
    # 加载模型 (自动处理 backbone 和 sdpa/flex attention)
    print(f"Loading model from {config.eval.checkpoint_path}...")
    model = diffusion_inf.Diffusion(config=config, tokenizer=tokenizer)
    model.to(device)
    model.eval()

    # 读取文件
    # input_file = "../text_io/demo_origin.txt"
    current_script_path = os.path.abspath(__file__)
    project_root = os.path.dirname(current_script_path)
    input_file = os.path.join(project_root, 'text_io', 'demo_origin.txt')
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