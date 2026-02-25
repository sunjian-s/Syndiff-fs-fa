import argparse
import torch
import torch.nn as nn
import numpy as np
import cv2
import os
import math
from backbones.ncsnpp_generator_adagn import NCSNpp 

# ==========================================
# 1. Diffusion 核心函数 (保持不变)
# ==========================================

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
    C = args.num_channels
    x = x_init[:, 0:C, :, :]
    source = x_init[:, C:, :, :]
    
    with torch.no_grad():
        for i in reversed(range(n_time)):
            t = torch.full((x.size(0),), i, dtype=torch.int64).to(x.device)
            t_time = t
            latent_z = torch.randn(x.size(0), args.nz, device=x.device)
            x_0 = generator(torch.cat((x, source), axis=1), t_time, latent_z)
            x_new = sample_posterior(coefficients, x_0[:, 0:C, :], x, t)
            x = x_new.detach()
    return x

# ==========================================
# 2. 增强版拼接与后处理
# ==========================================

def apply_clahe(img):
    """应用 CLAHE (限制对比度自适应直方图均衡化)"""
    # 转换到 LAB 色彩空间 (如果处理彩色) 或者直接处理灰度
    if len(img.shape) == 3:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl,a,b))
        final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    else:
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        final = clahe.apply(img)
    return final

def normalize_to_uint8(img_data):
    """
    智能拉伸对比度: 将 min-max 拉伸到 0-255
    """
    img_data = img_data - np.min(img_data)
    img_data = img_data / (np.max(img_data) + 1e-8)
    img_data = (img_data * 255.0).astype(np.uint8)
    return img_data

def get_gaussian_mask(size, sigma_scale=1/4):
    x = np.linspace(-1, 1, size)
    y = np.linspace(-1, 1, size)
    x, y = np.meshgrid(x, y)
    d = np.sqrt(x*x + y*y)
    sigma = sigma_scale 
    g = np.exp(-( (d)**2 / ( 2.0 * sigma**2 ) ) )
    return torch.from_numpy(g).float()

def predict_sliding_window(netG, image_path, args, device):
    print(f"正在处理: {image_path}")
    
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        print("错误：无法读取图片，请检查路径")
        return
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h_orig, w_orig, _ = img_rgb.shape
    
    # 预处理
    img_tensor = torch.from_numpy(img_rgb).float() / 127.5 - 1.0
    img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0).to(device)

    output_canvas = torch.zeros((1, 3, h_orig, w_orig), device=device)
    count_map = torch.zeros((1, 3, h_orig, w_orig), device=device)

    patch_size = args.image_size
    stride = int(patch_size * 0.5)
    
    gaussian_weight = get_gaussian_mask(patch_size).to(device)
    gaussian_weight = gaussian_weight.unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1)

    pos_coeff = Posterior_Coefficients(args, device)

    print(f"开始扫描... (尺寸: {h_orig}x{w_orig}, Patch: {patch_size}, Stride: {stride})")
    
    # 简单的覆盖式切片策略
    steps_h = list(range(0, h_orig - patch_size + 1, stride))
    if steps_h[-1] != h_orig - patch_size: steps_h.append(h_orig - patch_size)
    steps_w = list(range(0, w_orig - patch_size + 1, stride))
    if steps_w[-1] != w_orig - patch_size: steps_w.append(w_orig - patch_size)

    total_patches = len(steps_h) * len(steps_w)
    processed = 0
    
    # 保存一个中间 patch 用于调试
    debug_saved = False

    for y in steps_h:
        for x in steps_w:
            processed += 1
            print(f"\r  处理 Patch {processed}/{total_patches} ...", end="")
            
            source_patch = img_tensor[:, :, y:y+patch_size, x:x+patch_size]
            noise_patch = torch.randn_like(source_patch)
            x_init = torch.cat((noise_patch, source_patch), axis=1)

            # 预测
            predicted_patch = sample_from_model(pos_coeff, netG, args.num_timesteps, x_init, args)

            # --- 调试用：保存第一张 Patch 看看原始清晰度 ---
            if not debug_saved and processed == int(total_patches/2): # 取中间的一张
                debug_img = predicted_patch.squeeze().permute(1, 2, 0).cpu().numpy()
                debug_img = normalize_to_uint8(debug_img)
                debug_save_path = os.path.join(args.output_path, "debug_patch_raw.jpg")
                cv2.imwrite(debug_save_path, cv2.cvtColor(debug_img, cv2.COLOR_RGB2BGR))
                print(f"\n  [调试] 已保存原始 Patch 到: {debug_save_path} (请检查这张图是否清晰)")
                debug_saved = True
            # -------------------------------------------

            output_canvas[:, :, y:y+patch_size, x:x+patch_size] += predicted_patch * gaussian_weight
            count_map[:, :, y:y+patch_size, x:x+patch_size] += gaussian_weight

    print("\n扫描完成，正在融合...")
    
    count_map[count_map == 0] = 1.0
    final_output = output_canvas / count_map
    final_img = final_output.squeeze().permute(1, 2, 0).cpu().numpy()
    
    # 1. 基础结果 (Raw)
    img_raw = normalize_to_uint8(final_img)
    
    # 2. 增强结果 (CLAHE) - 专治血管不清晰
    print("正在应用 CLAHE 增强血管清晰度...")
    img_enhanced = apply_clahe(cv2.cvtColor(img_raw, cv2.COLOR_RGB2BGR)) # 转BGR给opencv处理

    # 保存
    filename = os.path.basename(image_path)
    
    path_raw = os.path.join(args.output_path, f"FFA_Raw_{filename}")
    path_enhanced = os.path.join(args.output_path, f"FFA_Enhanced_{filename}")
    
    cv2.imwrite(path_raw, cv2.cvtColor(img_raw, cv2.COLOR_RGB2BGR))
    cv2.imwrite(path_enhanced, img_enhanced)
    
    print(f"✅ 基础拼接结果: {path_raw}")
    print(f"✨ 清晰增强结果: {path_enhanced} (推荐看这张!)")


def main():
    parser = argparse.ArgumentParser()
    # 基础参数
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
    
    # 模型配置
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

    # 推理专用
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--input_image', type=str, required=True)
    parser.add_argument('--output_path', type=str, default='./inference_results')
    parser.add_argument('--gpu', type=int, default=0)

    args = parser.parse_args()

    device = torch.device(f'cuda:{args.gpu}')
    
    print(f"加载模型: {args.model_path}")
    netG = NCSNpp(args).to(device)
    
    state_dict = torch.load(args.model_path, map_location=device)
    if list(state_dict.keys())[0].startswith('module.'):
        new_state_dict = {k[7:]: v for k, v in state_dict.items()}
        state_dict = new_state_dict
    
    netG.load_state_dict(state_dict)
    netG.eval()

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    predict_sliding_window(netG, args.input_image, args, device)

if __name__ == '__main__':
    main()