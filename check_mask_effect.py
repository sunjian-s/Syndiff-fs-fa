import argparse
import torch
import torch.nn as nn
import numpy as np
import cv2
import os
import math
# 确保 backbones 文件夹在当前目录下
from backbones.ncsnpp_generator_adagn import NCSNpp 

# =========================================================================
# 1. Diffusion 核心数学函数
# =========================================================================

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
    C = args.num_channels # 这里应该是 4
    x = x_init[:, 0:C, :, :]
    source = x_init[:, C:, :, :]
    
    with torch.no_grad():
        for i in reversed(range(n_time)):
            t = torch.full((x.size(0),), i, dtype=torch.int64).to(x.device)
            t_time = t
            latent_z = torch.randn(x.size(0), args.nz, device=x.device)
            # 拼接输入：(噪声(4), 条件(4)) = 8通道
            x_0 = generator(torch.cat((x, source), axis=1), t_time, latent_z)
            # 后验采样
            x_new = sample_posterior(coefficients, x_0[:, 0:C, :], x, t)
            x = x_new.detach()
    return x

# =========================================================================
# 2. 诊断逻辑 (Diagnostic Logic)
# =========================================================================

def check_mask(netG, fundus_path, mask_path, args, device):
    print(f"🔎 正在诊断: {os.path.basename(fundus_path)}")
    print(f"  输入路径: {fundus_path}")
    
    if not os.path.exists(fundus_path):
        print(f"❌ 错误：找不到文件 -> {fundus_path}")
        return

    # 1. 读取并预处理输入 (Fundus + Mask)
    img_bgr = cv2.imread(fundus_path)
    if img_bgr is None:
        print("❌ 错误：OpenCV 无法读取 Fundus 图片")
        return
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]
    
    if not os.path.exists(mask_path):
         print(f"⚠️ 警告：Mask 文件不存在 -> {mask_path}")
         print("   -> 将使用全黑 Mask 代替")
         img_mask = np.zeros((h, w), dtype=np.uint8)
    else:
        img_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if img_mask is None:
            print("⚠️ 警告：Mask 文件存在但无法读取，使用全黑代替")
            img_mask = np.zeros((h, w), dtype=np.uint8)
    
    # 统一尺寸
    if img_mask.shape != (h, w):
        print(f"   -> 调整 Mask 尺寸: {img_mask.shape} -> {(h, w)}")
        img_mask = cv2.resize(img_mask, (w, h))

    # 归一化 (0-1)
    norm_fundus = img_rgb.astype(np.float32) / 255.0
    norm_mask = img_mask.astype(np.float32) / 255.0
    if norm_mask.ndim == 2:
        norm_mask = np.expand_dims(norm_mask, axis=-1)
    
    # 拼接成 4 通道输入 (H, W, 4)
    combined_input = np.concatenate((norm_fundus, norm_mask), axis=2) 
    # 转 Tensor (1, 4, H, W)
    input_tensor = torch.from_numpy(combined_input).float().permute(2, 0, 1).unsqueeze(0).to(device)
    
    # 2. 切出一个中心 Patch (256x256) 进行预测
    patch_size = args.image_size
    cy, cx = h // 2, w // 2
    y1 = max(0, cy - patch_size // 2)
    x1 = max(0, cx - patch_size // 2)
    y2, x2 = y1 + patch_size, x1 + patch_size
    
    # 确保不越界
    if y2 > h: y1, y2 = h - patch_size, h
    if x2 > w: x1, x2 = w - patch_size, w
    
    patch_input = input_tensor[:, :, y1:y2, x1:x2]
    
    # 归一化到 [-1, 1]
    patch_input = (patch_input * 2.0) - 1.0

    # 3. 构造模型输入
    noise = torch.randn_like(patch_input)
    x_init = torch.cat((noise, patch_input), axis=1) # (1, 8, 256, 256)
    
    # 4. 运行预测
    print("🤖 模型正在生成中...")
    pos_coeff = Posterior_Coefficients(args, device)
    output = sample_from_model(pos_coeff, netG, args.num_timesteps, x_init, args)
    
    # 5. 解析输出
    out_np = output.squeeze().permute(1, 2, 0).cpu().numpy()
    out_np = (out_np + 1.0) / 2.0 * 255.0
    out_np = np.clip(out_np, 0, 255).astype(np.uint8)
    
    gen_rgb = out_np[:, :, :3]   # 生成的血管图
    gen_mask = out_np[:, :, 3]   # 【关键】生成的 Mask 通道
    
    # 获取对应的输入真值
    gt_mask_patch = img_mask[y1:y2, x1:x2]
    
    # 6. 拼图可视化
    gen_mask_vis = cv2.cvtColor(gen_mask, cv2.COLOR_GRAY2BGR)
    gt_mask_vis = cv2.cvtColor(gt_mask_patch, cv2.COLOR_GRAY2BGR)
    gen_rgb_vis = cv2.cvtColor(gen_rgb, cv2.COLOR_RGB2BGR) # 转BGR保存
    
    cv2.putText(gen_rgb_vis, "Generated FFA", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(gen_mask_vis, "Generated Mask (Ch.4)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(gt_mask_vis, "Input GT Mask", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    final_comparison = np.hstack((gen_rgb_vis, gen_mask_vis, gt_mask_vis))
    
    save_name = os.path.join(args.output_path, "DIAGNOSE_MASK.jpg")
    cv2.imwrite(save_name, final_comparison)
    
    print("="*40)
    print(f"✅ 诊断图已保存: {save_name}")
    print("="*40)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--input_image', type=str, required=True)
    parser.add_argument('--mask_image', type=str, required=True)
    parser.add_argument('--output_path', type=str, default='./')
    parser.add_argument('--gpu', type=int, default=0)
    
    # 模型配置
    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--num_channels', type=int, default=4)
    parser.add_argument('--num_channels_dae', type=int, default=64)
    parser.add_argument('--ch_mult', nargs='+', type=int, default=[1, 1, 2, 2, 4, 4])
    parser.add_argument('--num_res_blocks', type=int, default=2)
    parser.add_argument('--num_timesteps', type=int, default=4)
    parser.add_argument('--nz', type=int, default=100)
    parser.add_argument('--z_emb_dim', type=int, default=256)
    parser.add_argument('--t_emb_dim', type=int, default=256)
    parser.add_argument('--n_mlp', type=int, default=3)
    
    # 其他默认参数
    parser.add_argument('--attn_resolutions', default=(16,))
    parser.add_argument('--dropout', type=float, default=0.)
    parser.add_argument('--resamp_with_conv', action='store_false', default=True)
    parser.add_argument('--conditional', action='store_false', default=True)
    parser.add_argument('--fir', action='store_false', default=True)
    parser.add_argument('--fir_kernel', default=[1, 3, 3, 1])
    parser.add_argument('--skip_rescale', action='store_false', default=True)
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
    parser.add_argument('--centered', action='store_false', default=True)

    args = parser.parse_args()

    device = torch.device(f'cuda:{args.gpu}')
    print(f"加载模型: {args.model_path}")
    netG = NCSNpp(args).to(device)
    state_dict = torch.load(args.model_path, map_location=device)
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k[7:]: v for k, v in state_dict.items()}
    netG.load_state_dict(state_dict)
    netG.eval()

    check_mask(netG, args.input_image, args.mask_image, args, device)

if __name__ == '__main__':
    main()