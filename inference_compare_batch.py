import argparse
import torch
import torch.nn as nn
import numpy as np
import cv2
import os
import glob
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from backbones.ncsnpp_generator_adagn import NCSNpp 

# ==========================================
# 1. Diffusion 核心函数
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
# 2. 批量处理逻辑
# ==========================================

def normalize_to_uint8(img_data):
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

def process_pair_with_mask(netG, fundus_path, ffa_path, mask_path, args, device, pos_coeff, gaussian_weight):
    filename = os.path.basename(fundus_path)
    print(f"处理: {filename} ...", end="", flush=True)
    
    # 1. 读取 Fundus
    img_bgr = cv2.imread(fundus_path)
    if img_bgr is None: return None, None
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h_orig, w_orig, _ = img_rgb.shape
    
    # 2. 读取 Mask (如果找不到，用全黑代替)
    if mask_path and os.path.exists(mask_path):
        img_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if img_mask.shape != (h_orig, w_orig):
            img_mask = cv2.resize(img_mask, (w_orig, h_orig))
    else:
        img_mask = np.zeros((h_orig, w_orig), dtype=np.uint8)

    if img_mask.ndim == 2:
        img_mask = np.expand_dims(img_mask, axis=-1)

    # 3. 构造 4 通道输入
    norm_fundus = img_rgb.astype(np.float32) / 255.0
    norm_mask = img_mask.astype(np.float32) / 255.0
    input_numpy = np.concatenate((norm_fundus, norm_mask), axis=2)
    
    input_tensor = torch.from_numpy(input_numpy).float()
    input_tensor = input_tensor.permute(2, 0, 1).unsqueeze(0).to(device)

    # 4. 读取真值 FFA
    target_bgr = cv2.imread(ffa_path)
    if target_bgr is None:
        target_rgb = None
    else:
        if target_bgr.shape[:2] != (h_orig, w_orig):
            target_bgr = cv2.resize(target_bgr, (w_orig, h_orig))
        target_rgb = cv2.cvtColor(target_bgr, cv2.COLOR_BGR2RGB)

    # 5. 滑动窗口生成
    output_canvas = torch.zeros((1, 4, h_orig, w_orig), device=device) # 4通道
    count_map = torch.zeros((1, 4, h_orig, w_orig), device=device)
    
    patch_size = args.image_size
    stride = args.stride if args.stride > 0 else int(patch_size * 0.5)
    
    steps_h = list(range(0, h_orig - patch_size + 1, stride))
    if steps_h[-1] != h_orig - patch_size: steps_h.append(h_orig - patch_size)
    steps_w = list(range(0, w_orig - patch_size + 1, stride))
    if steps_w[-1] != w_orig - patch_size: steps_w.append(w_orig - patch_size)

    # 扩展高斯权重到 4 通道
    g_weight = gaussian_weight.repeat(1, 4, 1, 1)

    for y in steps_h:
        for x in steps_w:
            source_patch = input_tensor[:, :, y:y+patch_size, x:x+patch_size]
            noise_patch = torch.randn_like(source_patch)
            x_init = torch.cat((noise_patch, source_patch), axis=1)
            predicted_patch = sample_from_model(pos_coeff, netG, args.num_timesteps, x_init, args)
            output_canvas[:, :, y:y+patch_size, x:x+patch_size] += predicted_patch * g_weight
            count_map[:, :, y:y+patch_size, x:x+patch_size] += g_weight

    count_map[count_map == 0] = 1.0
    final_output = output_canvas / count_map
    
    # 只取 RGB (前3通道)
    final_rgb = final_output[:, :3, :, :]
    final_img = final_rgb.squeeze().permute(1, 2, 0).cpu().numpy()
    fake_final = normalize_to_uint8(final_img)

    # 6. 计算指标
    val_psnr, val_ssim = 0.0, 0.0
    if target_rgb is not None:
        val_psnr = psnr(target_rgb, fake_final, data_range=255)
        try:
            val_ssim = ssim(target_rgb, fake_final, win_size=3, data_range=255, channel_axis=2)
        except:
            val_ssim = ssim(target_rgb, fake_final, win_size=3, data_range=255, multichannel=True)

    # 7. 拼图
    def put_text(img, text):
        img = img.copy()
        cv2.putText(img, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
        return img

    if target_rgb is not None:
        comparison = np.hstack((
            put_text(img_rgb, "Input+Mask"),
            put_text(fake_final, f"Gen (P:{val_psnr:.1f})"),
            put_text(target_rgb, "Ground Truth")
        ))
    else:
        comparison = np.hstack((
            put_text(img_rgb, "Input+Mask"),
            put_text(fake_final, "Generated")
        ))
    
    print(f" -> OK (PSNR: {val_psnr:.2f})")
    return comparison, (val_psnr, val_ssim)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--num_channels', type=int, default=4) # 默认4
    parser.add_argument('--num_channels_dae', type=int, default=64)
    parser.add_argument('--ch_mult', nargs='+', type=int, default=[1, 1, 2, 2, 4, 4])
    parser.add_argument('--num_res_blocks', type=int, default=2)
    parser.add_argument('--num_timesteps', type=int, default=4)
    parser.add_argument('--nz', type=int, default=100)
    parser.add_argument('--z_emb_dim', type=int, default=256)
    parser.add_argument('--t_emb_dim', type=int, default=256)
    parser.add_argument('--n_mlp', type=int, default=3)
    
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

    # 批量参数
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--input_dir', type=str, required=True)
    parser.add_argument('--target_dir', type=str, required=True)
    parser.add_argument('--mask_dir', type=str, required=True)
    parser.add_argument('--output_path', type=str, default='./inference_batch_results')
    parser.add_argument('--stride', type=int, default=32, help='滑动窗口步长 (越小越平滑但越慢)')
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

    pos_coeff = Posterior_Coefficients(args, device)
    patch_size = args.image_size
    gaussian_weight = get_gaussian_mask(patch_size).to(device)
    # 只需要 1 个通道，后面代码里会 repeat
    gaussian_weight = gaussian_weight.unsqueeze(0).unsqueeze(0)

    # 遍历
    exts = ('.jpg', '.jpeg', '.png', '.bmp')
    files = sorted([f for f in os.listdir(args.input_dir) if f.lower().endswith(exts)])
    print(f"找到 {len(files)} 张图片...")

    total_psnr = []
    total_ssim = []

    for f in files:
        fundus_path = os.path.join(args.input_dir, f)
        ffa_path = os.path.join(args.target_dir, f)
        
        # 尝试匹配 mask 文件名
        mask_path = os.path.join(args.mask_dir, f)
        if not os.path.exists(mask_path):
            name_no_ext = os.path.splitext(f)[0]
            # 尝试找 .png 或 .jpg
            mask_candidates = glob.glob(os.path.join(args.mask_dir, name_no_ext + ".*"))
            if mask_candidates: mask_path = mask_candidates[0]

        if not os.path.exists(ffa_path):
            print(f"跳过 {f} (无FFA真值)")
            continue

        comp_img, metrics = process_pair_with_mask(netG, fundus_path, ffa_path, mask_path, args, device, pos_coeff, gaussian_weight)
        
        if comp_img is not None:
            save_name = os.path.join(args.output_path, f"COMPARE_{f}")
            cv2.imwrite(save_name, cv2.cvtColor(comp_img, cv2.COLOR_RGB2BGR))
            total_psnr.append(metrics[0])
            total_ssim.append(metrics[1])

    if len(total_psnr) > 0:
        print(f"\n📊 平均 PSNR: {np.mean(total_psnr):.2f} dB")
        print(f"📊 平均 SSIM: {np.mean(total_ssim):.4f}")

if __name__ == '__main__':
    main()