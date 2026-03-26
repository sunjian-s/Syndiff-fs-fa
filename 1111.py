import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import os
import math
import lpips
import matplotlib.pyplot as plt
import random
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

# 引入你的模型定义
from backbones.ncsnpp_generator_adagn import NCSNpp 

# ==========================================
# 全局配置与工具函数
# ==========================================
def set_seed(seed=42):
    """固定随机种子以保证结果可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def var_func_vp(t, beta_min, beta_max):
    log_mean_coeff = -0.25 * t ** 2 * (beta_max - beta_min) - 0.5 * t * beta_min
    var = 1. - torch.exp(2. * log_mean_coeff)
    return var

def var_func_geometric(t, beta_min, beta_max):
    return beta_min * ((beta_max / beta_min) ** t)

def extract(input, t, shape):
    out = torch.gather(input, 0, t)
    reshape = [shape[0]] + [1] * (len(shape) - 1)
    out = out.reshape(*reshape)
    return out

def get_sigma_schedule(args, device):
    n_timestep = args.num_timesteps
    beta_min = args.beta_min
    beta_max = args.beta_max
    eps_small = 1e-3
    t = np.arange(0, n_timestep + 1, dtype=np.float64)
    t = t / n_timestep
    t = torch.from_numpy(t) * (1. - eps_small) + eps_small
    if args.use_geometric:
        var = var_func_geometric(t, beta_min, beta_max)
    else:
        var = var_func_vp(t, beta_min, beta_max)
    alpha_bars = 1.0 - var
    betas = 1 - alpha_bars[1:] / alpha_bars[:-1]
    first = torch.tensor(1e-8)
    betas = torch.cat((first[None], betas)).to(device)
    betas = betas.type(torch.float32)
    sigmas = betas**0.5
    a_s = torch.sqrt(1-betas)
    return sigmas, a_s, betas

class Posterior_Coefficients():
    def __init__(self, args, device):
        _, _, self.betas = get_sigma_schedule(args, device=device)
        self.betas = self.betas.type(torch.float32)[1:]
        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, 0)
        self.alphas_cumprod_prev = torch.cat(
            (torch.tensor([1.], dtype=torch.float32,device=device), self.alphas_cumprod[:-1]), 0
        )                               
        self.posterior_variance = self.betas * (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.rsqrt(self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1 / self.alphas_cumprod - 1)
        self.posterior_mean_coef1 = (self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1 - self.alphas_cumprod))
        self.posterior_mean_coef2 = ((1 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1 - self.alphas_cumprod))
        self.posterior_log_variance_clipped = torch.log(self.posterior_variance.clamp(min=1e-20))

def sample_posterior(coefficients, x_0, x_t, t):
    def q_posterior(x_0, x_t, t):
        mean = (
            extract(coefficients.posterior_mean_coef1, t, x_t.shape) * x_0
            + extract(coefficients.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        var = extract(coefficients.posterior_variance, t, x_t.shape)
        log_var_clipped = extract(coefficients.posterior_log_variance_clipped, t, x_t.shape)
        return mean, var, log_var_clipped
    
    def p_sample(x_0, x_t, t):
        mean, _, log_var = q_posterior(x_0, x_t, t)
        noise = torch.randn_like(x_t)
        nonzero_mask = (1 - (t == 0).type(torch.float32))
        return mean + nonzero_mask[:,None,None,None] * torch.exp(0.5 * log_var) * noise
            
    sample_x_pos = p_sample(x_0, x_t, t)
    return sample_x_pos

def sample_from_model(coefficients, generator, n_time, x_init, args):
    C = 3 
    x = x_init[:, 0:C, :, :]       # 噪声部分
    source = x_init[:, C:, :, :]   # 条件部分
    
    # 修复：latent_z 只生成一次，保证生成稳定
    latent_z = torch.randn(x.size(0), args.nz, device=x.device)
    
    with torch.no_grad():
        for i in reversed(range(n_time)):
            # 修复：timestep 转为 float32 类型，符合扩散模型要求
            t = torch.full((x.size(0),), i, dtype=torch.int64).to(x.device)
            t_time = t.to(torch.float32) / n_time  # 归一化到 [0,1]
            
            x_0 = generator(torch.cat((x, source), axis=1), t_time, latent_z)
            
            x_new = sample_posterior(coefficients, x_0[:, 0:C, :], x, t)
            x = x_new.detach()
    return x

def normalize_to_uint8(img_data):
    """
    安全的归一化函数：兼容 [-1,1] 和任意范围输入，转换为 [0,255] uint8
    """
    # 转换为 numpy 数组（兼容 tensor）
    if torch.is_tensor(img_data):
        img_data = img_data.cpu().numpy()
    
    # 情况1：输入是 [-1,1] 范围（训练时的归一化）
    if np.min(img_data) >= -1.0 + 1e-6 and np.max(img_data) <= 1.0 + 1e-6:
        img_data = (img_data + 1.0) / 2.0 * 255.0
    # 情况2：任意范围输入
    else:
        img_data = img_data - np.min(img_data)
        img_data = img_data / (np.max(img_data) + 1e-8) * 255.0
    
    # 截断越界值
    img_data = np.clip(img_data, 0, 255)
    return img_data.astype(np.uint8)

def get_hanning_mask(size, device):
    """生成 2D Hanning Window 用于平滑拼接（提前生成，避免冗余计算）"""
    window_1d = np.hanning(size)
    window_2d = np.outer(window_1d, window_1d)
    mask = torch.from_numpy(window_2d).float().to(device)
    # 扩展为 [1,3,H,W] 维度，匹配图片格式
    mask = mask.unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1)
    return mask

def predict_and_stitch(netG, fundus_path, ffa_path, args, device, lpips_model=None):
    print(f"\n🚀 开始处理: {os.path.basename(fundus_path)}")
    
    # 1. 读取并验证输入图片
    img_bgr = cv2.imread(fundus_path)
    if img_bgr is None:
        print(f"❌ 错误：找不到输入图片 {fundus_path}")
        return
    # 处理单通道图片（灰度图）
    if len(img_bgr.shape) == 2:
        img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2BGR)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h_orig, w_orig, c_orig = img_rgb.shape
    if c_orig != 3:
        print(f"❌ 错误：输入图片通道数必须为3，当前为{c_orig}")
        return
    
    # 归一化到 [-1, 1]
    img_tensor = torch.from_numpy(img_rgb).float() / 127.5 - 1.0
    img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0).to(device)

    # 2. 读取真值（Target）[可选]
    target_rgb = None
    if ffa_path and os.path.exists(ffa_path):
        target_bgr = cv2.imread(ffa_path)
        if target_bgr is None:
            print(f"⚠️ 警告：无法读取 Target 图片 {ffa_path}，将跳过指标计算")
        else:
            if len(target_bgr.shape) == 2:
                target_bgr = cv2.cvtColor(target_bgr, cv2.COLOR_GRAY2BGR)
            if target_bgr.shape[:2] != (h_orig, w_orig):
                target_bgr = cv2.resize(target_bgr, (w_orig, h_orig))
            target_rgb = cv2.cvtColor(target_bgr, cv2.COLOR_BGR2RGB)
    else:
        print("⚠️ 未提供 Target 图片或路径不存在，仅执行生成")

    # 3. 滑动窗口拼接（修复边界处理）
    patch_size = args.image_size
    # 处理图片尺寸小于 patch_size 的情况
    if h_orig < patch_size or w_orig < patch_size:
        print(f"⚠️ 图片尺寸({w_orig}x{h_orig})小于patch_size({patch_size})，将缩放图片")
        scale = max(patch_size / w_orig, patch_size / h_orig)
        new_w = int(w_orig * scale)
        new_h = int(h_orig * scale)
        img_tensor = F.interpolate(img_tensor, size=(new_h, new_w), mode='bilinear', align_corners=False)
        h_orig, w_orig = new_h, new_w
        if target_rgb is not None:
            target_rgb = cv2.resize(target_rgb, (new_w, new_h))
    
    stride = int(patch_size * 0.5) 
    output_canvas = torch.zeros((1, 3, h_orig, w_orig), device=device)
    count_map = torch.zeros((1, 3, h_orig, w_orig), device=device)
    weight_mask = get_hanning_mask(patch_size, device)

    pos_coeff = Posterior_Coefficients(args, device)

    # 修复滑动窗口边界计算
    steps_h = list(range(0, h_orig - patch_size + 1, stride))
    if steps_h and steps_h[-1] != h_orig - patch_size:
        steps_h.append(h_orig - patch_size)
    elif not steps_h:  # 图片刚好等于patch_size
        steps_h = [0]
    
    steps_w = list(range(0, w_orig - patch_size + 1, stride))
    if steps_w and steps_w[-1] != w_orig - patch_size:
        steps_w.append(w_orig - patch_size)
    elif not steps_w:
        steps_w = [0]

    total_patches = len(steps_h) * len(steps_w)
    processed = 0

    print(f"🧩 图像尺寸: {w_orig}x{h_orig}, Stride: {stride}, Total Patches: {total_patches}")
    
    for y in steps_h:
        for x in steps_w:
            processed += 1
            print(f"\r ⚡ 生成进度: {processed}/{total_patches} ...", end="", flush=True)
            
            source_patch = img_tensor[:, :, y:y+patch_size, x:x+patch_size]
            noise_patch = torch.randn_like(source_patch)
            x_init = torch.cat((noise_patch, source_patch), axis=1)
            
            predicted_patch = sample_from_model(pos_coeff, netG, args.num_timesteps, x_init, args)
            
            output_canvas[:, :, y:y+patch_size, x:x+patch_size] += predicted_patch * weight_mask
            count_map[:, :, y:y+patch_size, x:x+patch_size] += weight_mask

    # 修复：进度打印后换行
    print("\n✅ 融合完成。")
    
    # 4. 生成最终结果
    final_output = output_canvas / (count_map + 1e-8)
    # 还原到原始尺寸（如果之前缩放过）
    if final_output.shape[2:] != (img_rgb.shape[0], img_rgb.shape[1]):
        final_output = F.interpolate(final_output, size=img_rgb.shape[:2], mode='bilinear', align_corners=False)
    final_img = final_output.squeeze().permute(1, 2, 0)
    fake_final = normalize_to_uint8(final_img)

    # 5. 计算指标（优化精度）
    val_psnr = 0.0
    val_ssim = 0.0
    val_lpips = 0.0
    
    if target_rgb is not None:
        # 转换为float32计算，提升精度
        target_float = target_rgb.astype(np.float32) / 255.0
        fake_float = fake_final.astype(np.float32) / 255.0
        
        # A. PSNR
        val_psnr = psnr(target_float, fake_float, data_range=1.0)
        
        # B. SSIM
        try:
            val_ssim = ssim(target_float, fake_float, win_size=3, data_range=1.0, channel_axis=2)
        except Exception as e:
            val_ssim = ssim(target_float, fake_float, win_size=3, data_range=1.0, multichannel=True)
            print(f"⚠️ SSIM计算警告: {e}")
            
        # C. LPIPS（复用初始化的模型）
        try:
            if lpips_model is not None:
                # 归一化到 [-1, 1]
                target_tensor = torch.from_numpy(target_rgb).permute(2, 0, 1).unsqueeze(0).float().to(device)
                target_tensor = target_tensor / 127.5 - 1.0 
                fake_tensor = torch.from_numpy(fake_final).permute(2, 0, 1).unsqueeze(0).float().to(device)
                fake_tensor = fake_tensor / 127.5 - 1.0 
                
                with torch.no_grad():
                    val_lpips = lpips_model(fake_tensor, target_tensor).item()
        except Exception as e:
            print(f"⚠️ LPIPS 计算失败: {e}")

        print("="*30)
        print(f"📊 评估指标:")
        print(f"   PSNR:  {val_psnr:.4f} dB")
        print(f"   SSIM:  {val_ssim:.4f}")
        print(f"   LPIPS: {val_lpips:.4f} (越低越好)")
        print("="*30)

    # 6. 制作可视化大图
    def put_text(img, text, color=(255, 255, 255)):
        img = img.copy()
        # 适配不同图片尺寸的字体大小
        font_scale = max(0.5, min(img.shape[0], img.shape[1]) / 500)
        cv2.putText(img, text, (20, int(60 * font_scale)), 
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, max(1, int(3 * font_scale)))
        return img

    show_fundus = put_text(img_rgb, "Input (FS)")
    text_fake = "Ours (Generated)"
    if target_rgb is not None:
        text_fake = f"P:{val_psnr:.2f} S:{val_ssim:.3f} L:{val_lpips:.3f}"
    show_fake = put_text(fake_final, text_fake, color=(0, 255, 255))

    img_list = [show_fundus, show_fake]
    if target_rgb is not None:
        show_real = put_text(target_rgb, "Ground Truth")
        img_list.append(show_real)

    comparison = np.hstack(img_list)
    
    # 避免文件名覆盖
    filename = os.path.basename(fundus_path)
    name, ext = os.path.splitext(filename)
    save_path = os.path.join(args.output_path, f"RESULT_{name}{ext}")
    # 检查文件是否存在，避免覆盖
    counter = 1
    while os.path.exists(save_path):
        save_path = os.path.join(args.output_path, f"RESULT_{name}_{counter}{ext}")
        counter += 1
    
    # 保存结果
    cv2.imwrite(save_path, cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR))
    print(f"💾 结果对比图已保存: {save_path}")

def main():
    parser = argparse.ArgumentParser()
    # 基础配置
    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--num_channels', type=int, default=3)
    parser.add_argument('--num_channels_dae', type=int, default=64)
    parser.add_argument('--ch_mult', nargs='+', type=int, default=[1, 1, 2, 2, 4, 4])
    parser.add_argument('--num_res_blocks', type=int, default=2)
    parser.add_argument('--num_timesteps', type=int, default=4)
    parser.add_argument('--nz', type=int, default=100)
    parser.add_argument('--z_emb_dim', type=int, default=256)
    parser.add_argument('--t_emb_dim', type=int, default=256)
    parser.add_argument('--n_mlp', type=int, default=3)
    
    # Attention 参数
    parser.add_argument('--attn_resolutions', nargs='+', type=int, default=[16, 8])
    parser.add_argument('--local_attn_type', type=str, default='cbam')
    parser.add_argument('--local_attn_resolutions', nargs='+', type=int, default=[256, 128, 64])

    # 修复：argparse action 逻辑错误
    parser.add_argument('--dropout', type=float, default=0.)
    parser.add_argument('--resamp_with_conv', action='store_true', default=True)
    parser.add_argument('--conditional', action='store_true', default=True)
    parser.add_argument('--fir', action='store_true', default=True)
    parser.add_argument('--fir_kernel', default=[1, 3, 3, 1])
    parser.add_argument('--skip_rescale', action='store_true', default=True)
    parser.add_argument('--resblock_type', default='biggan')
    parser.add_argument('--progressive', type=str, default='none')
    parser.add_argument('--progressive_input', type=str, default='residual')
    parser.add_argument('--progressive_combine', type=str, default='sum')
    parser.add_argument('--embedding_type', type=str, default='positional')
    parser.add_argument('--fourier_scale', type=float, default=16.)
    parser.add_argument('--not_use_tanh', action='store_true', default=False)
    parser.add_argument('--beta_min', type=float, default=0.1)
    parser.add_argument('--beta_max', type=float, default=20.)
    parser.add_argument('--use_geometric', action='store_true', default=False)
    parser.add_argument('--centered', action='store_true', default=True)

    # 推理专用参数
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--input_image', type=str, required=True)
    parser.add_argument('--target_image', type=str, default=None)
    parser.add_argument('--output_path', type=str, default='./inference_results')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42, help='随机种子，保证结果可复现')
    parser.add_argument('--fp16', action='store_true', default=False, help='启用半精度推理，节省显存')

    args = parser.parse_args()

    # 固定随机种子
    set_seed(args.seed)

    # 设置设备
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"⚙️ 使用设备: {device}")
    
    # 创建输出目录（增加异常处理）
    try:
        os.makedirs(args.output_path, exist_ok=True)
        print(f"📂 输出目录: {args.output_path}")
    except Exception as e:
        print(f"❌ 创建输出目录失败: {e}")
        return
    
    print(f"📂 正在初始化 NCSNpp 模型...")
    netG = NCSNpp(args).to(device)
    
    # 启用半精度
    if args.fp16 and torch.cuda.is_available():
        netG = netG.half()
        print("⚡ 启用 FP16 半精度推理")
    
    print(f"📂 加载权重: {args.model_path}")
    try:
        checkpoint = torch.load(args.model_path, map_location=device)
        
        if 'gen_diffusive_2_dict' in checkpoint:
            state_dict = checkpoint['gen_diffusive_2_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
            
        if list(state_dict.keys())[0].startswith('module.'):
            new_state_dict = {k[7:]: v for k, v in state_dict.items()}
            state_dict = new_state_dict
            print("📌 已移除 module 前缀")
        
        netG.load_state_dict(state_dict, strict=True)
        print("✅ 权重加载成功！")
    except Exception as e:
        print(f"❌ 权重加载失败: {e}")
        return

    netG.eval()

    # 初始化 LPIPS 模型（只初始化一次，提升效率）
    lpips_model = None
    if args.target_image:
        try:
            lpips_model = lpips.LPIPS(net='alex').to(device)
            if args.fp16:
                lpips_model = lpips_model.half()
            lpips_model.eval()
        except Exception as e:
            print(f"⚠️ LPIPS 模型初始化失败: {e}")

    # 执行推理
    with torch.no_grad():
        predict_and_stitch(netG, args.input_image, args.target_image, args, device, lpips_model)

    # 清理显存
    torch.cuda.empty_cache()
    print("\n🎉 推理完成！")

if __name__ == '__main__':
    main()