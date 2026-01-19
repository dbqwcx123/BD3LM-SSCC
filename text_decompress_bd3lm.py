import os
import torch
import hydra
import logging
import numpy as np
import time
from tqdm import tqdm
from omegaconf import OmegaConf

import diffusion_inf
# 保持与你压缩脚本一致的导入路径
from code_utils import arithmetic_coder
from code_utils.ac_utils import normalize_pdf_for_arithmetic_coding

# --- Hydra Resolvers ---
try:
    OmegaConf.register_new_resolver('cwd', os.getcwd)
    OmegaConf.register_new_resolver('device_count', torch.cuda.device_count)
    OmegaConf.register_new_resolver('eval', eval)
    OmegaConf.register_new_resolver('div_up', lambda x, y: (x + y - 1) // y)
except:
    pass

def get_tokenizer(config):
    import transformers
    tokenizer = transformers.AutoTokenizer.from_pretrained(config.data.tokenizer_name_or_path)
    if tokenizer.bos_token is None and tokenizer.cls_token is not None:
        tokenizer.bos_token = tokenizer.cls_token
    if tokenizer.eos_token is None and tokenizer.sep_token is not None:
        tokenizer.eos_token = tokenizer.sep_token
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    return tokenizer

def get_confidence(log_probs, mask_indices):
    probs = torch.exp(log_probs)
    max_probs, _ = torch.max(probs, dim=-1) 
    confidences = max_probs[0, mask_indices]
    return confidences

def decompress_semi_ar(model, tokenizer, config, compressed_bits, original_length, device):
    """
    BD3LM 半自回归解压缩逻辑 (修复 Decoder 初始化)
    """
    # 1. 初始化参数
    block_size = model.block_size
    num_steps = config.algo.T
    context_size = config.model.length
    
    # 模拟 Padding 逻辑以确定总步数
    seqlen = original_length
    pad_len = (block_size - seqlen % block_size) % block_size
    total_len = seqlen + pad_len
    num_strides = total_len // block_size
    
    # --- [关键修改] 初始化算术解码器 ---
    # 1. 创建比特流迭代器
    # compressed_bits 是字符串 "010101...", 转为迭代器
    data_iter = iter(compressed_bits)
    
    # 2. 定义回调函数，每次返回一个整数 bit (0 或 1)
    def _input_fn(bit_sequence=data_iter):
        try:
            bit = next(bit_sequence)
            return int(bit)
        except StopIteration:
            return None
    
    # 3. 初始化 Decoder
    # 注意：这里必须与压缩时的 base 和 precision 保持一致 (压缩脚本中是 2 和 32)
    decoder = arithmetic_coder.Decoder(
        base=2, 
        precision=32,
        input_fn=_input_fn  # 使用回调函数
    )

    # 重置 KV Cache
    if config.sampling.kv_cache:
        model.backbone.reset_kv_cache(eval_batch_size=1)
    
    # x_accum 用于存储解压出的文本 (Context)
    x_accum = torch.tensor([], device=device, dtype=torch.long).reshape(1, 0)
    
    print(f"开始解压: 目标长度 {seqlen} (Pad后 {total_len}), 块大小 {block_size}")

    # 用于存放最终结果
    decoded_ids = []

    pbar = tqdm(range(num_strides))
    for stride_num in pbar:
        # --- 准备上下文 ---
        # 构造当前 Block 的初始状态 (全 Mask)
        current_block_masked = torch.full((1, block_size), model.mask_index, device=device, dtype=torch.long)
        x_accum = torch.cat((x_accum, current_block_masked), dim=1)
        
        # 滑动窗口
        end_idx = (stride_num + 1) * block_size
        start_idx = max(end_idx - context_size, 0)
        fwd_idx = torch.arange(start_idx, end_idx, device=device)
        
        x_input_window = x_accum[:, fwd_idx]

        # --- Block 内解压循环 ---
        maskable_mask = torch.zeros_like(x_input_window, dtype=torch.bool)
        maskable_mask[:, -block_size:] = True 
        
        timesteps = torch.linspace(1, 0, num_steps, device=device)
        
        for i in range(num_steps):
            current_mask_indices = torch.nonzero(maskable_mask[0]).flatten()
            num_current_masks = len(current_mask_indices)
            if num_current_masks == 0:
                break
                
            t = timesteps[i]
            _, move_chance_t = model.noise(t)
            sigma = model._sigma_from_p(move_chance_t).to(device)
            sigma = sigma.view(1, 1).repeat(x_input_window.shape[0], 1)

            with torch.no_grad():
                if config.sampling.kv_cache:
                    model_input = x_input_window[:, -block_size:]
                else:
                    model_input = x_input_window
                
                log_probs = model.forward(model_input, sigma, sample_mode=True, store_kv=False)
            
            # --- 确定性策略 (需与压缩端完全一致) ---
            if config.sampling.kv_cache:
                window_len = x_input_window.shape[1]
                offset = window_len - block_size
                rel_mask_indices = current_mask_indices - offset
                valid_mask = rel_mask_indices >= 0
                rel_mask_indices = rel_mask_indices[valid_mask]
                confidences = get_confidence(log_probs, rel_mask_indices)
            else:
                confidences = get_confidence(log_probs, current_mask_indices)

            sorted_indices = torch.argsort(confidences, descending=True)
            remaining_steps = num_steps - 1 - i
            ratio = 1.0 / (remaining_steps + 1)
            k = int(num_current_masks * ratio)
            k = max(1, min(k, num_current_masks))
            
            target_indices_global = current_mask_indices[sorted_indices[:k]]

            # --- 解码 (Inverse Operation) ---
            for global_idx in target_indices_global:
                global_idx = global_idx.item()
                
                # 获取概率分布
                if config.sampling.kv_cache:
                    rel_idx = global_idx - (x_input_window.shape[1] - block_size)
                    prob_dist = torch.exp(log_probs[0, rel_idx]).cpu().numpy()
                else:
                    prob_dist = torch.exp(log_probs[0, global_idx]).cpu().numpy()
                
                # [核心] 调用 Decoder 从比特流中还原 Token
                try:
                    decoded_token = decoder.decode(normalize_pdf_for_arithmetic_coding(prob_dist))
                except Exception as e:
                    print(f"解码异常 Block {stride_num}, Step {i}: {e}")
                    # 如果比特流耗尽或出错，可能抛出 StopIteration 或其他错误
                    # 这里为了防止程序崩掉，可以填入 UNK 或 Mask，但通常意味着数据损坏
                    decoded_token = tokenizer.unk_token_id if tokenizer.unk_token_id else 0

                # 更新状态
                x_input_window[0, global_idx] = decoded_token
                maskable_mask[0, global_idx] = False
                x_accum[0, start_idx + global_idx] = decoded_token

        # --- Block 完成，存入 Result ---
        decoded_block = x_accum[0, -block_size:].tolist()
        decoded_ids.extend(decoded_block)

        # --- 存储 KV (与压缩端一致) ---
        if config.sampling.kv_cache:
            with torch.no_grad():
                batch_size = x_accum.shape[0]
                sigma_zero = torch.zeros((batch_size, 1), device=device) 
                model_input = x_accum[:, -block_size:]
                model.forward(model_input, sigma_zero, sample_mode=True, store_kv=True)

    # 去除 Padding
    final_ids = decoded_ids[:original_length]
    text = tokenizer.decode(final_ids)
    return text

@hydra.main(version_base=None, config_path='configs', config_name='config_inf')
def main(config):
    torch.manual_seed(config.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = get_tokenizer(config)
    
    print(f"Loading model from {config.eval.checkpoint_path}...")
    model = diffusion_inf.Diffusion(config=config, tokenizer=tokenizer)
    model.to(device)
    model.eval()

    # 路径设置
    current_script_path = os.path.abspath(__file__)
    project_root = os.path.dirname(current_script_path)
    compressed_file = os.path.join(project_root, 'text_io', 'compressed_output.txt')
    
    if not os.path.exists(compressed_file):
        print(f"错误: 找不到压缩文件 {compressed_file}")
        # 如果没有压缩文件，尝试找个假数据或报错
        return

    # 读取压缩比特 (字符串形式)
    with open(compressed_file, "r") as f:
        compressed_bits = f.read().strip()
    
    print("-" * 30)
    # 获取原始长度 (必须正确，否则算术编码无法从流中正确截断)
    try:
        orig_len_input = input("请输入原始 Token 长度 (从 text_compress_bd3lm.py 的输出中获取): ")
        original_length = int(orig_len_input)
    except ValueError:
        print("输入无效，使用默认值 100 (可能导致解码错误)")
        original_length = 100
    
    start_time = time.time()
    try:
        recovered_text = decompress_semi_ar(model, tokenizer, config, compressed_bits, original_length, device)
    except Exception as e:
        print(f"解压过程中发生严重错误: {e}")
        import traceback
        traceback.print_exc()
        return

    end_time = time.time()
    
    print("\n" + "="*30)
    print(f"解压文本预览 (前100字符):\n{recovered_text[:100]}...")
    print(f"解压耗时: {end_time - start_time:.2f}s")
    print("="*30)
    
    output_file = os.path.join(project_root, 'text_io', 'decompressed_output.txt')
    with open(output_file, "w") as f:
        f.write(recovered_text)
    print(f"完整解压文本已保存至: {output_file}")

if __name__ == '__main__':
    main()