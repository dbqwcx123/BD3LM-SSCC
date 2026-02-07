import os
import torch
import torch.nn.functional as F
import hydra
import lightning as L
import numpy as np
import time
from contextlib import contextmanager
from tqdm import tqdm
from omegaconf import OmegaConf
from PIL import Image

import diffusion_inf
from code_utils import arithmetic_coder
from code_utils.ac_utils import normalize_pdf_for_arithmetic_coding
from code_utils.pixel_token_dict import compute_pixel_token_ids, tokenid_to_pixel
import utils
from models.hf import BD3LM, BD3LMConfig

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

# -------------------------- [Profiler Tool] --------------------------
class Profiler:
    def __init__(self):
        self.stats = {}
        
    @contextmanager
    def record(self, name):
        if torch.cuda.is_available(): torch.cuda.synchronize()
        start = time.perf_counter()
        yield
        if torch.cuda.is_available(): torch.cuda.synchronize()
        end = time.perf_counter()
        if name not in self.stats:
            self.stats[name] = []
        self.stats[name].append(end - start)

    def summary(self):
        print("\n---------- Decompression Profiling ----------")
        for name, times in self.stats.items():
            print(f"{name:<24} : {sum(times):.2f}s")
        print("------------------------------------------------------\n")

profiler = Profiler()

# -------------------- [Fast Vectorized Tokenizer] --------------------
class FastPixelTokenizer:
    """使用 Tensor 查找表替代字符串转换，加速 Token ID 转换"""
    def __init__(self, tokenizer, device):
        self.tokenizer = tokenizer
        self.device = device
        self.pixel_to_id = torch.zeros(256, dtype=torch.long, device=device)
        self.id_to_pixel = torch.zeros(tokenizer.vocab_size, dtype=torch.long, device=device)
        
        for i in range(256):
            token_id = tokenizer.convert_tokens_to_ids(str(i))
            self.pixel_to_id[i] = token_id
            self.id_to_pixel[token_id] = i

    def pixels_to_ids(self, pixel_tensor):
        return self.pixel_to_id[pixel_tensor.long()]

    def ids_to_pixels(self, id_tensor):
        return self.id_to_pixel[id_tensor.long()]

# --------------------- [Confidence Calculation] ----------------------
def get_confidence_entropy_batch(logits, mask_bool_matrix):
    """
    基于负熵计算置信度 - 向量化批量版
    logits: [B, Seq_Len, Vocab]
    mask_bool_matrix: [B, Seq_Len]
    """
    log_probs = F.log_softmax(logits, dim=-1)
    probs = torch.exp(log_probs)
    # Entropy: H(x) = - sum(p * log(p))
    entropy = -torch.sum(probs * log_probs, dim=-1)   # [B, Seq_Len]
    confidences = -entropy
    confidences = confidences.masked_fill(~mask_bool_matrix, float('-inf'))
    return confidences

# -------------------- [Arithmetic Decoder Input] ---------------------
def make_input_fn(bit_string):
    """构建算术解码器的输入流"""
    iterator = iter(bit_string)
    def _fn():
        try:
            return int(next(iterator))
        except StopIteration:
            return 0 
    return _fn

# ----------------------- [Batch Decompression] -----------------------
def decompress_batch(model, fast_tokenizer, config, pixel_token_ids_tensor, device, decoders, batch_size):
    """
    并行去噪解码核心函数
    """
    block_size = model.block_size
    num_steps = config.algo.T
    
    # 1. 初始化状态：全 Mask
    # [B, Block_Size]
    current_batch_masked = torch.full((batch_size, block_size), model.mask_index, device=device, dtype=torch.long)
    maskable_mask = torch.ones((batch_size, block_size), dtype=torch.bool, device=device)
    
    timesteps = torch.linspace(1, 0, num_steps, device=device)
    
    # 2. 扩散逆过程循环 (解码)
    for i in range(num_steps):
        if not maskable_mask.any():
            break
            
        t = timesteps[i]
        _, move_chance_t = model.noise(t)
        sigma = model._sigma_from_p(move_chance_t).to(device)
        sigma = sigma.view(1, 1).repeat(batch_size, 1)

        # A. Model Forward (Batch)
        with torch.no_grad(), profiler.record("model_forward"):
            raw_logits = model.forward(current_batch_masked, sigma, sample_mode=True, store_kv=False)
        
        # B. 计算置信度
        with profiler.record("confidence_calc"):
            logits_pixel = raw_logits[:, :, pixel_token_ids_tensor]  # [B, L, 256]
            confidences = get_confidence_entropy_batch(logits_pixel, maskable_mask)

        # C. 调度与采样 (Gather)
        with profiler.record("schedule_sample"):
            # 统计每个样本当前的 Mask 数量
            num_current_masks = maskable_mask.sum(dim=1)
            
            # 计算解耦的 K 值
            remaining_steps = num_steps - 1 - i
            ratio = 1.0 / (remaining_steps + 1)
            k_per_sample = (num_current_masks.float() * ratio).long()
            k_per_sample = torch.clamp(k_per_sample, min=1)
            k_per_sample = torch.min(k_per_sample, num_current_masks)

            # 获取排序后的索引 [B, L]
            sorted_indices = torch.argsort(confidences, descending=True, dim=1)
            
            # 准备概率分布
            all_probs = torch.softmax(logits_pixel.double(), dim=-1)  # [B, L, 256]

        # D. 算术解码与状态更新 (CPU Loop)
        with profiler.record("ac_decoding_update"):
            cpu_sorted_indices = sorted_indices.cpu()
            cpu_k_per_sample = k_per_sample.cpu()
            decoded_values_list = []
            
            # 对每个 Batch 独立处理
            for b in range(batch_size):
                k = cpu_k_per_sample[b].item()
                if k == 0: continue
                
                # 取出当前样本需要处理的 Top-K 位置
                target_indices = cpu_sorted_indices[b, :k]
                # 从 GPU Gather 对应的概率分布 [K, 256] -> CPU
                probs_b = all_probs[b, target_indices].cpu().numpy()
                
                for idx_in_k, seq_idx in enumerate(target_indices):
                    seq_idx = seq_idx.item()
                    
                    # 算术解码
                    try:
                        pixel_val = decoders[b].decode(
                            normalize_pdf_for_arithmetic_coding(probs_b[idx_in_k])
                        )
                        if not (0 <= pixel_val <= 255):
                            print(f"错误: 解码值 {pixel_val} 超出范围")
                            pixel_val = 128  # 沿用D3PM设置，解码失败则为灰像素
                    except Exception as e:
                        # print(f"解码错误 at batch {b}, idx {seq_idx}: {e}")
                        pixel_val = 128  # 沿用D3PM设置，解码失败则为灰像素
                        
                    decoded_values_list.append((b, seq_idx, pixel_val))
            
            # 批量更新 GPU Tensor
            if decoded_values_list:
                b_indices = [x[0] for x in decoded_values_list]
                s_indices = [x[1] for x in decoded_values_list]
                pixels = [x[2] for x in decoded_values_list]
                
                pixel_tensor = torch.tensor(pixels, device=device, dtype=torch.long)
                token_ids = fast_tokenizer.pixels_to_ids(pixel_tensor)
                
                current_batch_masked[b_indices, s_indices] = token_ids
                maskable_mask[b_indices, s_indices] = False

    return current_batch_masked


# ----------------------- [Reconstruction Helper] -----------------------
def reconstruct_image(patch_list, image_size, patch_size, is_channel_wised):
    """
    将 Patch 列表重组为图像
    """
    H = W = image_size
    P = patch_size
    num_patches_per_row = W // P
    patches_per_channel = (H // P) * (W // P)
    full_image = np.zeros((H, W, 3), dtype=np.uint8)
    
    if is_channel_wised:
        # 顺序：[All R] -> [All G] -> [All B]
        for c in range(3):
            c_offset = c * patches_per_channel
            for idx in range(patches_per_channel):
                row = idx // num_patches_per_row
                col = idx % num_patches_per_row
                
                if idx + c_offset >= len(patch_list): break
                p_data = patch_list[idx + c_offset]
                
                if p_data.size == P*P:
                    p_img = p_data.reshape(P, P)
                else:
                    print(f"Recon failed: p_data.shape{p_data.shape} mismatch ({P},{P})")
                
                full_image[row*P:(row+1)*P, col*P:(col+1)*P, c] = p_img
    else:
        # 顺序：Raster scan, RGB 通常在一个 patch 内处理 (flattened)
        for idx, p_data in enumerate(patch_list):
            row = idx // num_patches_per_row
            col = idx % num_patches_per_row
            
            if p_data.size == P*P*3:
                p_img = p_data.reshape(P, P, 3)
                full_image[row*P:(row+1)*P, col*P:(col+1)*P, :] = p_img
            elif p_data.size == P*P:
                p_img = p_data.reshape(P, P)
                full_image[row*P:(row+1)*P, col*P:(col+1)*P, :] = np.stack([p_img]*3, axis=-1)
            else:
                print(f"Recon failed: p_data.shape{p_data.shape} mismatch ({P},{P},3) or ({P},{P})")

    return full_image

# ----------------------- [Main Pipeline] -----------------------
def run_decompression_pipeline(model, tokenizer, config, device):
    input_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'image_io')
    input_file = os.path.join(input_path, f'compressed_output_bd3lm_opt.txt')
    output_dir = os.path.join(input_path, 'reconstructed_bd3lm')
    os.makedirs(output_dir, exist_ok=True)
    
    # 数据读取
    print(f"Reading compressed file: {input_file}")
    with open(input_file, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]
    
    # 准备组件
    fast_tokenizer = FastPixelTokenizer(tokenizer, device)
    pixel_token_ids_tensor = torch.tensor(compute_pixel_token_ids(tokenizer), device=device)
    
    # 配置参数
    batch_size = config.loader.eval_batch_size
    num_images = config.data.num_images_test
    img_size = config.data.image_size_test
    patch_size = config.data.patch_size
    is_channel_wised = config.data.is_channel_wised
    num_patches = config.data.num_patches_test // num_images
    num_patches_per_img = num_patches if is_channel_wised else num_patches//3
    patches_per_channel = np.square(img_size//patch_size)
    
    # 计算总行数是否符合预期
    total_lines = len(lines)
    assert total_lines == num_images
    
    print(f"Total Lines: {total_lines}")
    print(f"Image Size: {img_size}x{img_size}, Patch Size: {patch_size}")
    print(f"Patches Per Image: {num_patches_per_img} (Channel-Wised: {is_channel_wised})")
    print(f"Num Images: {num_images}")
    print(f"Decompression Batch Size (Parallel Images): {batch_size}")
    
    global_start_time = time.time()
    completed_images_count = 0
    
    # 3. 外层循环：按 Image Group 遍历
    lines_per_group = batch_size
    for group_start in range(0, total_lines, lines_per_group):
        group_end = min(group_start + lines_per_group, total_lines)
        group_lines = lines[group_start:group_end]
        current_bs = len(group_lines)
        if current_bs == 0: break
        print(f"Processing Image Group: {completed_images_count}-{completed_images_count + current_bs}...")
        
        # A. 重置 KV Cache (Image-Level Batching: 每一组新图开始时重置)
        model.backbone.reset_kv_cache(eval_batch_size=current_bs)
        
        # 初始化该 Batch 的 Decoders
        batch_pixel_buffers = [[] for _ in range(current_bs)]
        decoders = []
        for line in group_lines:
            # 移除 Stop Bit '1'
            if not line: bit_string = ""
            elif line[-1] == '1': bit_string = line[:-1]
            else: bit_string = line 
            
            decoders.append(arithmetic_coder.Decoder(
                base=config.data.ac_coder_base,
                precision=config.data.ac_coder_precision,
                input_fn=make_input_fn(bit_string)
            ))
        
        # B. 遍历 Patch (Stream Processing)
        for p_idx in tqdm(range(num_patches_per_img), desc="Decoding Patches"):
            # Channel Switch Reset Logic
            if is_channel_wised and p_idx > 0 and (p_idx % patches_per_channel == 0):
                 if config.sampling.reset_channel_context:
                     model.backbone.reset_kv_cache(eval_batch_size=current_bs)
            
            # --- 核心解压 ---
            # filled_tokens: [Batch, Block_Size]
            filled_tokens = decompress_batch(
                model, fast_tokenizer, config, 
                pixel_token_ids_tensor, device, decoders, current_bs
            )
            
            # --- 更新 KV Cache ---
            if p_idx < num_patches_per_img - 1: # 最后一步不需要
                with torch.no_grad(), profiler.record("kv_update"):
                    sigma_zero = torch.zeros((current_bs, 1), device=device)
                    _ = model.forward(filled_tokens, sigma_zero, sample_mode=True, store_kv=True)
            
            # --- 存储结果 ---
            # 将 Token ID 转回像素值
            pixels_tensor = fast_tokenizer.ids_to_pixels(filled_tokens) # [B, L]
            pixels_np = pixels_tensor.cpu().numpy().astype(np.uint8)
            for b in range(current_bs):
                batch_pixel_buffers[b].append(pixels_np[b])

        # C. 当前组图片解码完成，重组并保存
        print(f"Reconstructing {current_bs} images...")
        for b in range(current_bs):
            full_img = reconstruct_image(batch_pixel_buffers[b], img_size, patch_size, is_channel_wised)
            
            save_name = f"recon_{completed_images_count + b}.png"
            save_path = os.path.join(output_dir, save_name)
            Image.fromarray(full_img).save(save_path)
        completed_images_count += current_bs

    global_end_time = time.time()
    total_duration = global_end_time - global_start_time
    profiler.summary()
    
    print("="*30)
    print(f"Decompression Completed.")
    print(f"Total Images: {completed_images_count}")
    print(f"Total Time:   {total_duration:.4f} s")
    print(f"Results Saved to: {output_dir}")
    print("="*30)


def _load_from_checkpoint(config, tokenizer, device):
    if 'hf' in config.algo.backbone:
        return diffusion_inf.Diffusion(config, tokenizer=tokenizer).to(device)
    
    # 实例化最外层的 Diffusion 类
    # 此时 model.backbone 是一个原始的 DIT (由 Diffusion.__init__ 创建)
    model = diffusion_inf.Diffusion(config, tokenizer=tokenizer)
    
    # 构造中间层 BD3LM，并进行替换
    # push_to_hf 的逻辑
    bd3lm_config = BD3LMConfig(
        block_size=config.block_size,
        vocab_size=tokenizer.vocab_size+1,
        model_length=config.model.length,
        attn_backend=config.model.attn_backend,
        hidden_dim=768,
        cond_dim=128,
        n_blocks=12,
        n_heads=12,
        dropout=0.1,
        time_conditioning=False,
        return_dict=False
    )
    bd3lm_backbone = BD3LM(bd3lm_config)
    
    # 将 Diffusion 的 backbone 替换为 BD3LM
    # Diffusion(model) -> BD3LM(model.backbone) -> DIT(model.backbone.backbone)
    model.backbone = bd3lm_backbone
    
    ckpt = torch.load(config.sampling.checkpoint_path, map_location='cpu', weights_only=False)
    state_dict = ckpt['state_dict']
    
    model = model.to(device)
    model.load_state_dict(state_dict, strict=False)
    ema_params = ckpt['ema']['shadow_params']
    for s_param, param in zip(ema_params, model.parameters()):
        if param.requires_grad:
            param.data.copy_(s_param.data)

    return model


@hydra.main(version_base=None, config_path='configs', config_name='config_inf')
def main(config):
    L.seed_everything(config.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = utils.get_tokenizer(config)
    
    print(f"Loading model from {config.sampling.checkpoint_path}...")
    model = _load_from_checkpoint(config, tokenizer, device)
    model.to(device)
    model.backbone.eval()
    model.noise.eval()
    
    # torch.set_float32_matmul_precision('high')
    
    run_decompression_pipeline(model, tokenizer, config, device)

if __name__ == '__main__':
    main()