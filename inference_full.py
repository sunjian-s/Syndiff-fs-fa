import argparse
import torch
import torch.nn as nn
import numpy as np
import cv2
import os
import math
# 确保你的目录下有 backbones 文件夹
from backbones.ncsnpp_generator_adagn import NCSNpp 

# ==========================================
# 1. 必要的 Diffusion 辅助函数
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
    # 动态切片：前C通道是噪声，后C通道是条件
    C = args.num_channels
    x = x_init[:, 0:C, :, :]
    source = x_init[:, C:, :, :]
    
    with torch.no_grad():
        for i in reversed(range(n_time)):
            t = torch.full((x.size(0),), i, dtype=torch.int64).to(x.device)
            t_time = t
            latent_z = torch.randn(x.size(0), args.nz, device=x.device)
            # 拼接输入：(噪声, 条件)
            x_0 = generator(torch.cat((x, source), axis=1), t_time, latent_z)
            # 后验采样
            x_new = sample_posterior(coefficients, x_0[:, 0:C, :], x, t)
            x = x_new.detach()
    return x

# ==========================================
# 2. 滑动窗口拼接逻辑
# ==========================================

def get_gaussian_mask(size, sigma_scale=1/4):
    """生成二维高斯掩码，中心为1，边缘衰减"""
    x = np.linspace(-1, 1, size)
    y = np.linspace(-1, 1, size)
    x, y = np.meshgrid(x, y)
    d = np.sqrt(x*x + y*y)
    sigma = sigma_scale 
    g = np.exp(-( (d)**2 / ( 2.0 * sigma**2 ) ) )
    return torch.from_numpy(g).float()

def predict_sliding_window(netG, image_path, args, device):
    print(f"正在处理: {image_path}")
    
    # 1. 读取并预处理图片
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        print("错误：无法读取图片，请检查路径是否正确")
        return
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h_orig, w_orig, _ = img_rgb.shape
    
    # 归一化到 [-1, 1] 并转为 Tensor
    img_tensor = torch.from_numpy(img_rgb).float() / 127.5 - 1.0
    img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0).to(device) # (1, 3, H, W)

    # 2. 初始化两张画布：一张存累加结果，一张存权重计数
    output_canvas = torch.zeros((1, 3, h_orig, w_orig), device=device)
    count_map = torch.zeros((1, 3, h_orig, w_orig), device=device)

    # 3. 准备参数
    patch_size = args.image_size
    stride = int(patch_size * 0.5) # 50% 重叠
    
    # 准备高斯权重 (让拼接更平滑)
    gaussian_weight = get_gaussian_mask(patch_size).to(device)
    # 扩展到3通道 (1, 3, 256, 256)
    gaussian_weight = gaussian_weight.unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1)

    # 准备采样系数
    pos_coeff = Posterior_Coefficients(args, device)

    # 4. 滑动窗口扫描
    print(f"开始扫描... (尺寸: {h_orig}x{w_orig}, Patch: {patch_size}, Stride: {stride})")
    
    # 计算切片起始点列表
    steps_h = list(range(0, h_orig - patch_size + 1, stride))
    if steps_h[-1] != h_orig - patch_size: steps_h.append(h_orig - patch_size)
    
    steps_w = list(range(0, w_orig - patch_size + 1, stride))
    if steps_w[-1] != w_orig - patch_size: steps_w.append(w_orig - patch_size)

    total_patches = len(steps_h) * len(steps_w)
    processed = 0

    for y in steps_h:
        for x in steps_w:
            processed += 1
            print(f"\r  处理 Patch {processed}/{total_patches} ...", end="")
            
            # (A) 切出一块 (Source)
            source_patch = img_tensor[:, :, y:y+patch_size, x:x+patch_size]
            
            # (B) 构造输入：生成纯噪声作为 Target 的初始状态
            noise_patch = torch.randn_like(source_patch)
            
            # 这里的 x_init 是 (B, 6, H, W) -> 前3通道噪声，后3通道条件
            x_init = torch.cat((noise_patch, source_patch), axis=1)

            # (C) 喂给模型预测
            predicted_patch = sample_from_model(pos_coeff, netG, args.num_timesteps, x_init, args)

            # (D) 累加回画布
            output_canvas[:, :, y:y+patch_size, x:x+patch_size] += predicted_patch * gaussian_weight
            count_map[:, :, y:y+patch_size, x:x+patch_size] += gaussian_weight

    print("\n扫描完成，正在融合...")
    
    # 5. 取加权平均
    count_map[count_map == 0] = 1.0 # 避免除以0
    final_output = output_canvas / count_map
    
    # 6. 后处理与保存
    final_img = final_output.squeeze().permute(1, 2, 0).cpu().numpy()
    # 反归一化 [-1, 1] -> [0, 255]
    final_img = (final_img + 1.0) / 2.0 * 255.0
    final_img = np.clip(final_img, 0, 255).astype(np.uint8)
    
    # 保存
    filename = os.path.basename(image_path)
    save_name = os.path.join(args.output_path, f"FULL_FFA_{filename}")
    # 转回 BGR 保存
    cv2.imwrite(save_name, cv2.cvtColor(final_img, cv2.COLOR_RGB2BGR))
    print(f"✅ 结果已保存: {save_name}")


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
    
    # 【关键修复】添加 n_mlp 参数，防止报错
    parser.add_argument('--n_mlp', type=int, default=3, help='number of mlp layers for z')
    
    # 模型配置参数 (为了兼容 NCSNpp 初始化)
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

    # 推理专用参数
    parser.add_argument('--model_path', type=str, required=True, help='训练好的 .pth 权重文件路径')
    parser.add_argument('--input_image', type=str, required=True, help='要转换的 Fundus 大图路径')
    parser.add_argument('--output_path', type=str, default='./inference_results', help='输出文件夹')
    parser.add_argument('--gpu', type=int, default=0)

    args = parser.parse_args()

    device = torch.device(f'cuda:{args.gpu}')
    
    print(f"加载模型: {args.model_path}")
    netG = NCSNpp(args).to(device)
    
    # 处理 DDP 权重 (移除 module. 前缀)
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