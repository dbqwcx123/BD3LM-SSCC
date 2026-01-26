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

import diffusion_inf
from code_utils import arithmetic_coder
from code_utils.ac_utils import normalize_pdf_for_arithmetic_coding
from code_utils.pixel_token_dict import compute_pixel_token_ids, tokenid_to_pixel
import utils
import data_loaders 
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

# ------------------------- [Profiling Tool] --------------------------
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
        print("\n---------- Compression Profiling ----------")
        for name, times in self.stats.items():
            print(f"{name:<24} : {sum(times):.2f}s")
        print("-------------------------------------------\n")

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
    entropy = -torch.sum(probs * log_probs, dim=-1)  # [B, Seq_Len]
    confidences = -entropy
    confidences = confidences.masked_fill(~mask_bool_matrix, float('-inf'))
    return confidences

# ------------------------ [Batch Compression] ------------------------
def compress_batch(model, fast_tokenizer, config, current_block_ids_batch, pixel_token_ids_tensor, device, encoders):
    """
    并行去噪编码核心函数
    """
    batch_size, block_size = current_block_ids_batch.shape
    num_steps = config.algo.T
    
    # 1. 初始化状态：全 Mask
    current_batch_masked = torch.full((batch_size, block_size), model.mask_index, device=device, dtype=torch.long)
    maskable_mask = torch.ones((batch_size, block_size), dtype=torch.bool, device=device)
    
    timesteps = torch.linspace(1, 0, num_steps, device=device)
    
    # 2. 扩散逆过程循环 (编码)
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

        # D. 算术编码与状态更新 (CPU Loop, GPU Data Prep)
        with profiler.record("ac_encoding_update"):
            cpu_sorted_indices = sorted_indices.cpu()
            cpu_k_per_sample = k_per_sample.cpu()
            cpu_true_blocks = current_block_ids_batch.cpu()
            
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
                    true_token_id = cpu_true_blocks[b, seq_idx].item()
                    true_pixel_value = fast_tokenizer.id_to_pixel[true_token_id].item()
                    
                    # 算术编码
                    encoders[b].encode(
                        normalize_pdf_for_arithmetic_coding(probs_b[idx_in_k]), 
                        true_pixel_value
                    )
                    
                    # 更新状态
                    current_batch_masked[b, seq_idx] = true_token_id
                    maskable_mask[b, seq_idx] = False

    return current_batch_masked


def run_compression_pipeline(model, tokenizer, config, data_iterator, device):
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'image_io')
    os.makedirs(output_path, exist_ok=True)
    output_file = os.path.join(output_path, f'compressed_output_bd3lm_opt.txt')
    with open(output_file, "w") as f: pass
    
    # 数据读取与重组 (Custom Image-Level Batching Logic)
    # 将 iterator 转为 list (按图分组)，以便 Batch 处理
    all_images_patches = [] 
    current_img = []
    prev_fid = -1
    
    for data, fid in tqdm(data_iterator, desc="Loading"):
        if prev_fid != -1 and fid != prev_fid:
            all_images_patches.append(current_img)
            current_img = []
        current_img.append(data)
        prev_fid = fid
    if current_img: all_images_patches.append(current_img)
    
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
    
    print(f"Image Size: {img_size}x{img_size}, Patch Size: {patch_size}")
    print(f"Patches Per Image: {num_patches_per_img} (Channel-Wised: {is_channel_wised})")
    print(f"Num Images: {num_images}")
    print(f"Compression Batch Size (Parallel Images): {batch_size}")
    
    global_start_time = time.time()
    total_bits, total_pixels = 0, 0
    
    # 3. 外层循环：按 Batch 处理图片组
    for b_start in range(0, num_images, batch_size):
        b_end = min(b_start + batch_size, num_images)
        current_batch_imgs = all_images_patches[b_start:b_end]
        current_bs = len(current_batch_imgs)
        print(f"Processing Image Batch {b_start}-{b_end}...")
        
        # A. 重置 KV Cache (Image-Level Batching: 每一组新图开始时重置)
        model.backbone.reset_kv_cache(eval_batch_size=current_bs)
        
        # 初始化该 Batch 的 Encoders
        batch_bit_buffers = [[] for _ in range(current_bs)]
        encoders = [
            arithmetic_coder.Encoder(
                base=config.data.ac_coder_base,
                precision=config.data.ac_coder_precision,
                output_fn=batch_bit_buffers[i].append
            ) for i in range(current_bs)
        ]
        
        # B. 遍历 Patch (Stream Processing)
        for p_idx in tqdm(range(num_patches_per_img), desc="Encoding Patches"):
            # Channel Switch Reset Logic
            if is_channel_wised and p_idx > 0 and (p_idx % patches_per_channel == 0):
                if config.sampling.reset_channel_context:
                    model.backbone.reset_kv_cache(eval_batch_size=current_bs)
            
            # 构造 Patch Batch [B, Seq_Len]
            patch_data_list = [img[p_idx].flatten() for img in current_batch_imgs]
            patch_tensor = torch.tensor(np.stack(patch_data_list), device=device)
            # Pixel -> Token ID (Vectorized)
            current_block_ids_batch = fast_tokenizer.pixels_to_ids(patch_tensor)
            
            # --- 核心压缩 ---
            # filled_block_tensor: [Batch, Block_Size]
            filled_block_tensor = compress_batch(
                model, fast_tokenizer, config, current_block_ids_batch, 
                pixel_token_ids_tensor, device, encoders
            )
            
            # --- 更新 KV Cache ---
            if p_idx < num_patches_per_img - 1: # 最后一步不需要
                with torch.no_grad(), profiler.record("kv_cache_update"):
                    sigma_zero = torch.zeros((current_bs, 1), device=device)
                    _ = model.forward(filled_block_tensor, sigma_zero, sample_mode=True, store_kv=True)

        # C. 当前组图片编码完成，写入并保存
        with profiler.record("write_bits"):
            for i in range(current_bs):
                encoders[i].terminate()
                bits = "".join(map(str, batch_bit_buffers[i]))
                bits += '1' # Stop bit
                
                total_bits += len(bits)
                total_pixels += num_patches_per_img * model.block_size
                
                with open(output_file, "a") as f:
                    f.write(bits + '\n')
    
    global_end_time = time.time()
    total_duration = global_end_time - global_start_time
    profiler.summary()

    print("="*30)
    print(f"Compression Completed.")
    print(f"Total Bits: {total_bits}")
    print(f"Total Time: {total_duration:.4f} s")
    print(f"Average BPP: {total_bits/total_pixels:.4f}" if total_pixels > 0 else "BPP: N/A")
    print(f"Result Saved to: {output_file}")
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
    
    # 1. 加载模型
    print(f"Loading model from {config.sampling.checkpoint_path}...")
    model = _load_from_checkpoint(config, tokenizer, device)
    model.to(device)
    model.backbone.eval()
    model.noise.eval()
    
    # torch.set_float32_matmul_precision('high')
    
    # 2. 加载数据
    if config.data.test_dataset == "CIFAR10":
        test_dataset_path = os.path.join(config.data.dataset_root, "CIFAR10", "cifar10_test")
    elif config.data.test_dataset  == "DIV2K":
        test_dataset_path  = os.path.join(config.data.dataset_root, "DIV2K", "DIV2K_LR_unified", "X4", "test")
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