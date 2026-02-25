# # import argparse
# # import torch
# # import torch.nn as nn
# # import numpy as np
# # import cv2
# # import os
# # import math
# # from skimage.metrics import peak_signal_noise_ratio as psnr
# # from skimage.metrics import structural_similarity as ssim

# # # 引入你的模型定义 (确保 backbones 文件夹在同级目录)
# # from backbones.ncsnpp_generator_adagn import NCSNpp 

# # # ==========================================
# # # 1. Diffusion 核心工具函数 (保持与训练一致)
# # # ==========================================

# # def var_func_vp(t, beta_min, beta_max):
# #     log_mean_coeff = -0.25 * t ** 2 * (beta_max - beta_min) - 0.5 * t * beta_min
# #     var = 1. - torch.exp(2. * log_mean_coeff)
# #     return var

# # def var_func_geometric(t, beta_min, beta_max):
# #     return beta_min * ((beta_max / beta_min) ** t)

# # def extract(input, t, shape):
# #     out = torch.gather(input, 0, t)
# #     reshape = [shape[0]] + [1] * (len(shape) - 1)
# #     out = out.reshape(*reshape)
# #     return out

# # def get_sigma_schedule(args, device):
# #     n_timestep = args.num_timesteps
# #     beta_min = args.beta_min
# #     beta_max = args.beta_max
# #     eps_small = 1e-3
# #     t = np.arange(0, n_timestep + 1, dtype=np.float64)
# #     t = t / n_timestep
# #     t = torch.from_numpy(t) * (1. - eps_small) + eps_small
# #     if args.use_geometric:
# #         var = var_func_geometric(t, beta_min, beta_max)
# #     else:
# #         var = var_func_vp(t, beta_min, beta_max)
# #     alpha_bars = 1.0 - var
# #     betas = 1 - alpha_bars[1:] / alpha_bars[:-1]
# #     first = torch.tensor(1e-8)
# #     betas = torch.cat((first[None], betas)).to(device)
# #     betas = betas.type(torch.float32)
# #     sigmas = betas**0.5
# #     a_s = torch.sqrt(1-betas)
# #     return sigmas, a_s, betas

# # class Posterior_Coefficients():
# #     def __init__(self, args, device):
# #         _, _, self.betas = get_sigma_schedule(args, device=device)
# #         self.betas = self.betas.type(torch.float32)[1:]
# #         self.alphas = 1 - self.betas
# #         self.alphas_cumprod = torch.cumprod(self.alphas, 0)
# #         self.alphas_cumprod_prev = torch.cat(
# #             (torch.tensor([1.], dtype=torch.float32,device=device), self.alphas_cumprod[:-1]), 0
# #         )              
# #         self.posterior_variance = self.betas * (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod)
# #         self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
# #         self.sqrt_recip_alphas_cumprod = torch.rsqrt(self.alphas_cumprod)
# #         self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1 / self.alphas_cumprod - 1)
# #         self.posterior_mean_coef1 = (self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1 - self.alphas_cumprod))
# #         self.posterior_mean_coef2 = ((1 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1 - self.alphas_cumprod))
# #         self.posterior_log_variance_clipped = torch.log(self.posterior_variance.clamp(min=1e-20))

# # def sample_posterior(coefficients, x_0, x_t, t):
# #     def q_posterior(x_0, x_t, t):
# #         mean = (
# #             extract(coefficients.posterior_mean_coef1, t, x_t.shape) * x_0
# #             + extract(coefficients.posterior_mean_coef2, t, x_t.shape) * x_t
# #         )
# #         var = extract(coefficients.posterior_variance, t, x_t.shape)
# #         log_var_clipped = extract(coefficients.posterior_log_variance_clipped, t, x_t.shape)
# #         return mean, var, log_var_clipped
    
# #     def p_sample(x_0, x_t, t):
# #         mean, _, log_var = q_posterior(x_0, x_t, t)
# #         noise = torch.randn_like(x_t)
# #         nonzero_mask = (1 - (t == 0).type(torch.float32))
# #         return mean + nonzero_mask[:,None,None,None] * torch.exp(0.5 * log_var) * noise
            
# #     sample_x_pos = p_sample(x_0, x_t, t)
# #     return sample_x_pos

# # def sample_from_model(coefficients, generator, n_time, x_init, args):
# #     # 【强制3通道】确保这里切分正确
# #     C = 3 # 强制为3，忽略 args.num_channels 的潜在错误
# #     x = x_init[:, 0:C, :, :]       # 噪声部分 (前3通道)
# #     source = x_init[:, C:, :, :]   # 条件部分 (后3通道，即输入图片)
    
# #     with torch.no_grad():
# #         for i in reversed(range(n_time)):
# #             t = torch.full((x.size(0),), i, dtype=torch.int64).to(x.device)
# #             t_time = t
# #             latent_z = torch.randn(x.size(0), args.nz, device=x.device)
            
# #             # 输入给模型的必须是 6 通道 (3噪声 + 3条件)
# #             x_0 = generator(torch.cat((x, source), axis=1), t_time, latent_z)
            
# #             x_new = sample_posterior(coefficients, x_0[:, 0:C, :], x, t)
# #             x = x_new.detach()
# #     return x

# # # ==========================================
# # # 2. 图像处理与拼接逻辑 (优化版：消除网格线)
# # # ==========================================

# # def normalize_to_uint8(img_data):
# #     """将 [-1, 1] 或任意范围的 tensor/numpy 归一化到 [0, 255] uint8"""
# #     img_data = img_data - np.min(img_data)
# #     img_data = img_data / (np.max(img_data) + 1e-8)
# #     img_data = (img_data * 255.0).astype(np.uint8)
# #     return img_data

# # def get_hanning_mask(size):
# #     """
# #     【改进】使用 2D Hanning Window。
# #     相比高斯窗，Hanning 窗在边缘严格收敛为 0，能更好地消除拼接痕迹。
# #     """
# #     # 生成 1D 汉宁窗
# #     window_1d = np.hanning(size)
# #     # 外积生成 2D 权重图
# #     window_2d = np.outer(window_1d, window_1d)
# #     return torch.from_numpy(window_2d).float()

# # def predict_and_stitch(netG, fundus_path, ffa_path, args, device):
# #     print(f"\n🚀 开始处理 (高精度融合模式): {os.path.basename(fundus_path)}")
    
# #     # 1. 读取输入 (Fundus - Source)
# #     img_bgr = cv2.imread(fundus_path)
# #     if img_bgr is None:
# #         print(f"❌ 错误：找不到输入图片 {fundus_path}")
# #         return
# #     img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
# #     h_orig, w_orig, _ = img_rgb.shape
    
# #     # 归一化到 [-1, 1] 并转 Tensor
# #     img_tensor = torch.from_numpy(img_rgb).float() / 127.5 - 1.0
# #     img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0).to(device) # [1, 3, H, W]

# #     # 2. 读取真值 (Real FFA - Target) [可选]
# #     target_rgb = None
# #     if ffa_path and os.path.exists(ffa_path):
# #         target_bgr = cv2.imread(ffa_path)
# #         # 强制调整真值尺寸以匹配输入
# #         if target_bgr.shape[:2] != (h_orig, w_orig):
# #             target_bgr = cv2.resize(target_bgr, (w_orig, h_orig))
# #         target_rgb = cv2.cvtColor(target_bgr, cv2.COLOR_BGR2RGB)

# #     # 3. 滑动窗口拼接 (Stitching)
# #     patch_size = args.image_size
    
# #     # 【关键改进 1】步长设为 patch_size 的 50%
# #     # 50% 的重叠率是消除边缘痕迹的最佳平衡点
# #     stride = int(patch_size * 0.5) 
    
# #     # 初始化大画布
# #     output_canvas = torch.zeros((1, 3, h_orig, w_orig), device=device)
# #     count_map = torch.zeros((1, 3, h_orig, w_orig), device=device)
    
# #     # 【关键改进 2】使用 Hanning Mask 替代高斯
# #     weight_mask = get_hanning_mask(patch_size).to(device)
# #     weight_mask = weight_mask.unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1) # [1, 3, 256, 256]
    
# #     pos_coeff = Posterior_Coefficients(args, device)

# #     # 计算切片坐标 (确保覆盖最右侧和最下侧)
# #     steps_h = list(range(0, h_orig - patch_size + 1, stride))
# #     if steps_h[-1] != h_orig - patch_size: 
# #         steps_h.append(h_orig - patch_size)
    
# #     steps_w = list(range(0, w_orig - patch_size + 1, stride))
# #     if steps_w[-1] != w_orig - patch_size: 
# #         steps_w.append(w_orig - patch_size)

# #     total_patches = len(steps_h) * len(steps_w)
# #     processed = 0

# #     print(f"🧩 图像尺寸: {w_orig}x{h_orig}, Stride: {stride}, Total Patches: {total_patches}")
    
# #     for y in steps_h:
# #         for x in steps_w:
# #             processed += 1
# #             print(f"\r  ⚡ 生成进度: {processed}/{total_patches} ...", end="")
            
# #             # 提取 patch
# #             source_patch = img_tensor[:, :, y:y+patch_size, x:x+patch_size]
            
# #             # 生成噪声
# #             noise_patch = torch.randn_like(source_patch)
            
# #             # 构造输入: [Noise, Source] -> 6 通道
# #             x_init = torch.cat((noise_patch, source_patch), axis=1)
            
# #             # 扩散模型预测
# #             predicted_patch = sample_from_model(pos_coeff, netG, args.num_timesteps, x_init, args)
            
# #             # 【核心逻辑】加权融合
# #             # 预测值 * 权重 (边缘权重近乎0，中心权重1)
# #             output_canvas[:, :, y:y+patch_size, x:x+patch_size] += predicted_patch * weight_mask
# #             # 记录权重累加值
# #             count_map[:, :, y:y+patch_size, x:x+patch_size] += weight_mask

# #     print("\n✅ 融合完成。")
    
# #     # 归一化 (总像素值 / 总权重值)
# #     # 这里加 1e-8 防止除以 0，虽然 Hanning 窗叠加后很难为 0
# #     final_output = output_canvas / (count_map + 1e-8)
    
# #     # 转回 CPU numpy
# #     final_img = final_output.squeeze().permute(1, 2, 0).cpu().numpy()
    
# #     # 转换格式用于保存
# #     fake_final = normalize_to_uint8(final_img)

# #     # 4. 计算指标 (如果有真值)
# #     val_psnr = 0.0
# #     val_ssim = 0.0
# #     if target_rgb is not None:
# #         val_psnr = psnr(target_rgb, fake_final, data_range=255)
# #         try:
# #             val_ssim = ssim(target_rgb, fake_final, win_size=3, data_range=255, channel_axis=2)
# #         except:
# #             val_ssim = ssim(target_rgb, fake_final, win_size=3, data_range=255, multichannel=True)
            
# #         print("="*30)
# #         print(f"📊 评估指标:")
# #         print(f"   PSNR: {val_psnr:.4f} dB")
# #         print(f"   SSIM: {val_ssim:.4f}")
# #         print("="*30)

# #     # 5. 制作对比大图
# #     def put_text(img, text):
# #         img = img.copy()
# #         # 字体稍微调小一点，防止遮挡
# #         cv2.putText(img, text, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
# #         return img

# #     show_fundus = put_text(img_rgb, "Input (FS)")
    
# #     text_fake = f"Ours (FA)"
# #     if target_rgb is not None:
# #         text_fake += f" PSNR:{val_psnr:.2f}"
# #     show_fake = put_text(fake_final, text_fake)

# #     # 拼接列表
# #     img_list = [show_fundus, show_fake]
    
# #     if target_rgb is not None:
# #         show_real = put_text(target_rgb, "Ground Truth (FA)")
# #         img_list.append(show_real)

# #     # 横向拼接
# #     comparison = np.hstack(img_list)
    
# #     # 保存结果
# #     filename = os.path.basename(fundus_path)
# #     save_path = os.path.join(args.output_path, f"RESULT_SMOOTH_{filename}")
    
# #     # OpenCV 使用 BGR，所以要转换
# #     cv2.imwrite(save_path, cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR))
# #     print(f"💾 结果已保存: {save_path}")


# # def main():
# #     parser = argparse.ArgumentParser()
# #     # 基础配置 (必须与训练一致)
# #     parser.add_argument('--image_size', type=int, default=256)
# #     parser.add_argument('--num_channels', type=int, default=3, help='强制为3，对应RGB')
# #     parser.add_argument('--num_channels_dae', type=int, default=64)
# #     parser.add_argument('--ch_mult', nargs='+', type=int, default=[1, 1, 2, 2, 4, 4])
# #     parser.add_argument('--num_res_blocks', type=int, default=2)
# #     parser.add_argument('--num_timesteps', type=int, default=4)
# #     parser.add_argument('--nz', type=int, default=100)
# #     parser.add_argument('--z_emb_dim', type=int, default=256)
# #     parser.add_argument('--t_emb_dim', type=int, default=256)
# #     parser.add_argument('--n_mlp', type=int, default=3)
    
# #     # NCSNpp 参数
# #     parser.add_argument('--attn_resolutions', default=(16,))
# #     parser.add_argument('--dropout', type=float, default=0.)
# #     parser.add_argument('--resamp_with_conv', action='store_false', default=True)
# #     parser.add_argument('--conditional', action='store_false', default=True)
# #     parser.add_argument('--fir', action='store_false', default=True)
# #     parser.add_argument('--fir_kernel', default=[1, 3, 3, 1])
# #     parser.add_argument('--skip_rescale', action='store_false', default=True)
# #     parser.add_argument('--resblock_type', default='biggan')
# #     parser.add_argument('--progressive', type=str, default='none')
# #     parser.add_argument('--progressive_input', type=str, default='residual')
# #     parser.add_argument('--progressive_combine', type=str, default='sum')
# #     parser.add_argument('--embedding_type', type=str, default='positional')
# #     parser.add_argument('--fourier_scale', type=float, default=16.)
# #     parser.add_argument('--not_use_tanh', action='store_true', default=False)
# #     parser.add_argument('--beta_min', type=float, default=0.1)
# #     parser.add_argument('--beta_max', type=float, default=20.)
# #     parser.add_argument('--use_geometric', action='store_true', default=False)
# #     parser.add_argument('--centered', action='store_false', default=True)

# #     # 推理专用参数
# #     parser.add_argument('--model_path', type=str, required=True, help='content.pth 或 gen_diffusive_2_x.pth 的路径')
# #     parser.add_argument('--input_image', type=str, required=True, help='输入 Fundus 大图路径')
# #     parser.add_argument('--target_image', type=str, default=None, help='(可选) 真实 FFA 大图路径')
# #     parser.add_argument('--output_path', type=str, default='./inference_results', help='结果保存文件夹')
# #     parser.add_argument('--gpu', type=int, default=0)

# #     args = parser.parse_args()

# #     device = torch.device(f'cuda:{args.gpu}')
    
# #     print(f"📂 加载模型权重: {args.model_path}")
# #     # 初始化模型
# #     netG = NCSNpp(args).to(device)
    
# #     # 智能加载权重 (处理 content.pth 字典结构)
# #     checkpoint = torch.load(args.model_path, map_location=device)
    
# #     if 'gen_diffusive_2_dict' in checkpoint:
# #         # Case 1: 加载的是 content.pth (包含所有)
# #         print("检测到完整 checkpoint，正在提取 gen_diffusive_2 (FS->FA)...")
# #         state_dict = checkpoint['gen_diffusive_2_dict']
# #     elif 'state_dict' in checkpoint:
# #         state_dict = checkpoint['state_dict']
# #     else:
# #         # Case 2: 加载的是单独的权重文件
# #         state_dict = checkpoint
        
# #     # 处理 module. 前缀 (如果是 DDP 训练保存的)
# #     if list(state_dict.keys())[0].startswith('module.'):
# #         new_state_dict = {k[7:]: v for k, v in state_dict.items()}
# #         state_dict = new_state_dict
    
# #     netG.load_state_dict(state_dict)
# #     netG.eval()

# #     if not os.path.exists(args.output_path):
# #         os.makedirs(args.output_path)

# #     # 执行拼接推理
# #     predict_and_stitch(netG, args.input_image, args.target_image, args, device)

# # if __name__ == '__main__':
# #     main()
# import argparse  # <--- 已补上
# import torch
# import torch.nn as nn
# import numpy as np
# import cv2
# import os
# import math
# from skimage.metrics import peak_signal_noise_ratio as psnr
# from skimage.metrics import structural_similarity as ssim

# # 引入你的模型定义 (确保 backbones 文件夹在同级目录)
# # 假设你的目录结构是:
# # ├── inference.py
# # └── backbones/
# #     └── ncsnpp_generator_adagn.py
# from backbones.ncsnpp_generator_adagn import NCSNpp 

# # ==========================================
# # 1. Diffusion 核心工具函数 (保持与训练一致)
# # ==========================================

# def var_func_vp(t, beta_min, beta_max):
#     log_mean_coeff = -0.25 * t ** 2 * (beta_max - beta_min) - 0.5 * t * beta_min
#     var = 1. - torch.exp(2. * log_mean_coeff)
#     return var

# def var_func_geometric(t, beta_min, beta_max):
#     return beta_min * ((beta_max / beta_min) ** t)

# def extract(input, t, shape):
#     out = torch.gather(input, 0, t)
#     reshape = [shape[0]] + [1] * (len(shape) - 1)
#     out = out.reshape(*reshape)
#     return out

# def get_sigma_schedule(args, device):
#     n_timestep = args.num_timesteps
#     beta_min = args.beta_min
#     beta_max = args.beta_max
#     eps_small = 1e-3
#     t = np.arange(0, n_timestep + 1, dtype=np.float64)
#     t = t / n_timestep
#     t = torch.from_numpy(t) * (1. - eps_small) + eps_small
#     if args.use_geometric:
#         var = var_func_geometric(t, beta_min, beta_max)
#     else:
#         var = var_func_vp(t, beta_min, beta_max)
#     alpha_bars = 1.0 - var
#     betas = 1 - alpha_bars[1:] / alpha_bars[:-1]
#     first = torch.tensor(1e-8)
#     betas = torch.cat((first[None], betas)).to(device)
#     betas = betas.type(torch.float32)
#     sigmas = betas**0.5
#     a_s = torch.sqrt(1-betas)
#     return sigmas, a_s, betas

# class Posterior_Coefficients():
#     def __init__(self, args, device):
#         _, _, self.betas = get_sigma_schedule(args, device=device)
#         self.betas = self.betas.type(torch.float32)[1:]
#         self.alphas = 1 - self.betas
#         self.alphas_cumprod = torch.cumprod(self.alphas, 0)
#         self.alphas_cumprod_prev = torch.cat(
#             (torch.tensor([1.], dtype=torch.float32,device=device), self.alphas_cumprod[:-1]), 0
#         )                      
#         self.posterior_variance = self.betas * (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod)
#         self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
#         self.sqrt_recip_alphas_cumprod = torch.rsqrt(self.alphas_cumprod)
#         self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1 / self.alphas_cumprod - 1)
#         self.posterior_mean_coef1 = (self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1 - self.alphas_cumprod))
#         self.posterior_mean_coef2 = ((1 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1 - self.alphas_cumprod))
#         self.posterior_log_variance_clipped = torch.log(self.posterior_variance.clamp(min=1e-20))

# def sample_posterior(coefficients, x_0, x_t, t):
#     def q_posterior(x_0, x_t, t):
#         mean = (
#             extract(coefficients.posterior_mean_coef1, t, x_t.shape) * x_0
#             + extract(coefficients.posterior_mean_coef2, t, x_t.shape) * x_t
#         )
#         var = extract(coefficients.posterior_variance, t, x_t.shape)
#         log_var_clipped = extract(coefficients.posterior_log_variance_clipped, t, x_t.shape)
#         return mean, var, log_var_clipped
    
#     def p_sample(x_0, x_t, t):
#         mean, _, log_var = q_posterior(x_0, x_t, t)
#         noise = torch.randn_like(x_t)
#         nonzero_mask = (1 - (t == 0).type(torch.float32))
#         return mean + nonzero_mask[:,None,None,None] * torch.exp(0.5 * log_var) * noise
            
#     sample_x_pos = p_sample(x_0, x_t, t)
#     return sample_x_pos

# def sample_from_model(coefficients, generator, n_time, x_init, args):
#     # 【强制3通道】确保这里切分正确
#     C = 3 # 强制为3，忽略 args.num_channels 的潜在错误
#     x = x_init[:, 0:C, :, :]       # 噪声部分 (前3通道)
#     source = x_init[:, C:, :, :]   # 条件部分 (后3通道，即输入图片)
    
#     with torch.no_grad():
#         for i in reversed(range(n_time)):
#             t = torch.full((x.size(0),), i, dtype=torch.int64).to(x.device)
#             t_time = t
#             latent_z = torch.randn(x.size(0), args.nz, device=x.device)
            
#             # 输入给模型的必须是 6 通道 (3噪声 + 3条件)
#             x_0 = generator(torch.cat((x, source), axis=1), t_time, latent_z)
            
#             x_new = sample_posterior(coefficients, x_0[:, 0:C, :], x, t)
#             x = x_new.detach()
#     return x

# # ==========================================
# # 2. 图像处理与拼接逻辑 (优化版：消除网格线)
# # ==========================================

# def normalize_to_uint8(img_data):
#     """将 [-1, 1] 或任意范围的 tensor/numpy 归一化到 [0, 255] uint8"""
#     img_data = img_data - np.min(img_data)
#     img_data = img_data / (np.max(img_data) + 1e-8)
#     img_data = (img_data * 255.0).astype(np.uint8)
#     return img_data

# def get_hanning_mask(size):
#     """
#     【改进】使用 2D Hanning Window。
#     相比高斯窗，Hanning 窗在边缘严格收敛为 0，能更好地消除拼接痕迹。
#     """
#     # 生成 1D 汉宁窗
#     window_1d = np.hanning(size)
#     # 外积生成 2D 权重图
#     window_2d = np.outer(window_1d, window_1d)
#     return torch.from_numpy(window_2d).float()

# def predict_and_stitch(netG, fundus_path, ffa_path, args, device):
#     print(f"\n🚀 开始处理 (高精度融合模式): {os.path.basename(fundus_path)}")
    
#     # 1. 读取输入 (Fundus - Source)
#     img_bgr = cv2.imread(fundus_path)
#     if img_bgr is None:
#         print(f"❌ 错误：找不到输入图片 {fundus_path}")
#         return
#     img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
#     h_orig, w_orig, _ = img_rgb.shape
    
#     # 归一化到 [-1, 1] 并转 Tensor
#     img_tensor = torch.from_numpy(img_rgb).float() / 127.5 - 1.0
#     img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0).to(device) # [1, 3, H, W]

#     # 2. 读取真值 (Real FFA - Target) [可选]
#     target_rgb = None
#     if ffa_path and os.path.exists(ffa_path):
#         target_bgr = cv2.imread(ffa_path)
#         # 强制调整真值尺寸以匹配输入
#         if target_bgr.shape[:2] != (h_orig, w_orig):
#             target_bgr = cv2.resize(target_bgr, (w_orig, h_orig))
#         target_rgb = cv2.cvtColor(target_bgr, cv2.COLOR_BGR2RGB)

#     # 3. 滑动窗口拼接 (Stitching)
#     patch_size = args.image_size
    
#     # 【关键改进 1】步长设为 patch_size 的 50%
#     stride = int(patch_size * 0.5) 
    
#     # 初始化大画布
#     output_canvas = torch.zeros((1, 3, h_orig, w_orig), device=device)
#     count_map = torch.zeros((1, 3, h_orig, w_orig), device=device)
    
#     # 【关键改进 2】使用 Hanning Mask 替代高斯
#     weight_mask = get_hanning_mask(patch_size).to(device)
#     weight_mask = weight_mask.unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1) # [1, 3, 256, 256]
    
#     pos_coeff = Posterior_Coefficients(args, device)

#     # 计算切片坐标 (确保覆盖最右侧和最下侧)
#     steps_h = list(range(0, h_orig - patch_size + 1, stride))
#     if steps_h[-1] != h_orig - patch_size: 
#         steps_h.append(h_orig - patch_size)
    
#     steps_w = list(range(0, w_orig - patch_size + 1, stride))
#     if steps_w[-1] != w_orig - patch_size: 
#         steps_w.append(w_orig - patch_size)

#     total_patches = len(steps_h) * len(steps_w)
#     processed = 0

#     print(f"🧩 图像尺寸: {w_orig}x{h_orig}, Stride: {stride}, Total Patches: {total_patches}")
    
#     for y in steps_h:
#         for x in steps_w:
#             processed += 1
#             print(f"\r  ⚡ 生成进度: {processed}/{total_patches} ...", end="")
            
#             # 提取 patch
#             source_patch = img_tensor[:, :, y:y+patch_size, x:x+patch_size]
            
#             # 生成噪声
#             noise_patch = torch.randn_like(source_patch)
            
#             # 构造输入: [Noise, Source] -> 6 通道
#             x_init = torch.cat((noise_patch, source_patch), axis=1)
            
#             # 扩散模型预测
#             predicted_patch = sample_from_model(pos_coeff, netG, args.num_timesteps, x_init, args)
            
#             # 【核心逻辑】加权融合
#             output_canvas[:, :, y:y+patch_size, x:x+patch_size] += predicted_patch * weight_mask
#             # 记录权重累加值
#             count_map[:, :, y:y+patch_size, x:x+patch_size] += weight_mask

#     print("\n✅ 融合完成。")
    
#     # 归一化 (总像素值 / 总权重值)
#     final_output = output_canvas / (count_map + 1e-8)
    
#     # 转回 CPU numpy
#     final_img = final_output.squeeze().permute(1, 2, 0).cpu().numpy()
    
#     # 转换格式用于保存
#     fake_final = normalize_to_uint8(final_img)

#     # 4. 计算指标 (如果有真值)
#     val_psnr = 0.0
#     val_ssim = 0.0
#     if target_rgb is not None:
#         val_psnr = psnr(target_rgb, fake_final, data_range=255)
#         try:
#             val_ssim = ssim(target_rgb, fake_final, win_size=3, data_range=255, channel_axis=2)
#         except:
#             val_ssim = ssim(target_rgb, fake_final, win_size=3, data_range=255, multichannel=True)
            
#         print("="*30)
#         print(f"📊 评估指标:")
#         print(f"   PSNR: {val_psnr:.4f} dB")
#         print(f"   SSIM: {val_ssim:.4f}")
#         print("="*30)

#     # 5. 制作对比大图
#     def put_text(img, text):
#         img = img.copy()
#         cv2.putText(img, text, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
#         return img

#     show_fundus = put_text(img_rgb, "Input (FS)")
    
#     text_fake = f"Ours (FA)"
#     if target_rgb is not None:
#         text_fake += f" PSNR:{val_psnr:.2f}"
#     show_fake = put_text(fake_final, text_fake)

#     # 拼接列表
#     img_list = [show_fundus, show_fake]
    
#     if target_rgb is not None:
#         show_real = put_text(target_rgb, "Ground Truth (FA)")
#         img_list.append(show_real)

#     # 横向拼接
#     comparison = np.hstack(img_list)
    
#     # 保存结果
#     filename = os.path.basename(fundus_path)
#     save_path = os.path.join(args.output_path, f"RESULT_SMOOTH_{filename}")
    
#     cv2.imwrite(save_path, cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR))
#     print(f"💾 结果已保存: {save_path}")


# def main():
#     parser = argparse.ArgumentParser()
#     # 基础配置 (必须与训练一致)
#     parser.add_argument('--image_size', type=int, default=256)
#     parser.add_argument('--num_channels', type=int, default=3, help='强制为3，对应RGB')
#     parser.add_argument('--num_channels_dae', type=int, default=64)
#     parser.add_argument('--ch_mult', nargs='+', type=int, default=[1, 1, 2, 2, 4, 4])
#     parser.add_argument('--num_res_blocks', type=int, default=2)
#     parser.add_argument('--num_timesteps', type=int, default=4)
#     parser.add_argument('--nz', type=int, default=100)
#     parser.add_argument('--z_emb_dim', type=int, default=256)
#     parser.add_argument('--t_emb_dim', type=int, default=256)
#     parser.add_argument('--n_mlp', type=int, default=3)
    
#     # NCSNpp 参数
#     parser.add_argument('--attn_resolutions', default=(16,))
#     parser.add_argument('--dropout', type=float, default=0.)
#     parser.add_argument('--resamp_with_conv', action='store_false', default=True)
#     parser.add_argument('--conditional', action='store_false', default=True)
#     parser.add_argument('--fir', action='store_false', default=True)
#     parser.add_argument('--fir_kernel', default=[1, 3, 3, 1])
#     parser.add_argument('--skip_rescale', action='store_false', default=True)
#     parser.add_argument('--resblock_type', default='biggan')
#     parser.add_argument('--progressive', type=str, default='none')
#     parser.add_argument('--progressive_input', type=str, default='residual')
#     parser.add_argument('--progressive_combine', type=str, default='sum')
#     parser.add_argument('--embedding_type', type=str, default='positional')
#     parser.add_argument('--fourier_scale', type=float, default=16.)
#     parser.add_argument('--not_use_tanh', action='store_true', default=False)
#     parser.add_argument('--beta_min', type=float, default=0.1)
#     parser.add_argument('--beta_max', type=float, default=20.)
#     parser.add_argument('--use_geometric', action='store_true', default=False)
#     parser.add_argument('--centered', action='store_false', default=True)

#     # 推理专用参数
#     parser.add_argument('--model_path', type=str, required=True, help='content.pth 或 gen_diffusive_2_x.pth 的路径')
#     parser.add_argument('--input_image', type=str, required=True, help='输入 Fundus 大图路径')
#     parser.add_argument('--target_image', type=str, default=None, help='(可选) 真实 FFA 大图路径')
#     parser.add_argument('--output_path', type=str, default='./inference_results', help='结果保存文件夹')
#     parser.add_argument('--gpu', type=int, default=0)

#     args = parser.parse_args()

#     device = torch.device(f'cuda:{args.gpu}')
    
#     print(f"📂 加载模型权重: {args.model_path}")
#     # 初始化模型
#     netG = NCSNpp(args).to(device)
    
#     # 智能加载权重 (处理 content.pth 字典结构)
#     checkpoint = torch.load(args.model_path, map_location=device)
    
#     if 'gen_diffusive_2_dict' in checkpoint:
#         # Case 1: 加载的是 content.pth (包含所有)
#         print("检测到完整 checkpoint，正在提取 gen_diffusive_2 (FS->FA)...")
#         state_dict = checkpoint['gen_diffusive_2_dict']
#     elif 'state_dict' in checkpoint:
#         state_dict = checkpoint['state_dict']
#     else:
#         # Case 2: 加载的是单独的权重文件
#         state_dict = checkpoint
        
#     # 处理 module. 前缀 (如果是 DDP 训练保存的)
#     if list(state_dict.keys())[0].startswith('module.'):
#         new_state_dict = {k[7:]: v for k, v in state_dict.items()}
#         state_dict = new_state_dict
    
#     netG.load_state_dict(state_dict)
#     netG.eval()

#     if not os.path.exists(args.output_path):
#         os.makedirs(args.output_path)

#     # 执行拼接推理
#     predict_and_stitch(netG, args.input_image, args.target_image, args, device)

# if __name__ == '__main__':
#     main()
import argparse
import torch
import torch.nn as nn
import numpy as np
import cv2
import os
import math
import lpips  # <--- 【新增】引入 LPIPS
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

# 引入你的模型定义
from backbones.ncsnpp_generator_adagn import NCSNpp 

# ==========================================
# 1. Diffusion 核心工具函数
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
    C = 3 
    x = x_init[:, 0:C, :, :]       # 噪声部分
    source = x_init[:, C:, :, :]   # 条件部分
    
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
# 2. 图像处理与拼接逻辑
# ==========================================

def normalize_to_uint8(img_data):
    """将 [-1, 1] 或任意范围的 tensor/numpy 归一化到 [0, 255] uint8"""
    img_data = img_data - np.min(img_data)
    img_data = img_data / (np.max(img_data) + 1e-8)
    img_data = (img_data * 255.0).astype(np.uint8)
    return img_data

def get_hanning_mask(size):
    """生成 2D Hanning Window 用于平滑拼接"""
    window_1d = np.hanning(size)
    window_2d = np.outer(window_1d, window_1d)
    return torch.from_numpy(window_2d).float()

def predict_and_stitch(netG, fundus_path, ffa_path, args, device):
    print(f"\n🚀 开始处理 (高精度融合模式): {os.path.basename(fundus_path)}")
    
    # 1. 读取输入 (Fundus)
    img_bgr = cv2.imread(fundus_path)
    if img_bgr is None:
        print(f"❌ 错误：找不到输入图片 {fundus_path}")
        return
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h_orig, w_orig, _ = img_rgb.shape
    
    # 归一化到 [-1, 1]
    img_tensor = torch.from_numpy(img_rgb).float() / 127.5 - 1.0
    img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0).to(device)

    # 2. 读取真值 (Target) [可选]
    target_rgb = None
    if ffa_path and os.path.exists(ffa_path):
        target_bgr = cv2.imread(ffa_path)
        if target_bgr.shape[:2] != (h_orig, w_orig):
            target_bgr = cv2.resize(target_bgr, (w_orig, h_orig))
        target_rgb = cv2.cvtColor(target_bgr, cv2.COLOR_BGR2RGB)

    

    # 3. 滑动窗口拼接
    patch_size = args.image_size
    stride = int(patch_size * 0.5) 
    
    output_canvas = torch.zeros((1, 3, h_orig, w_orig), device=device)
    count_map = torch.zeros((1, 3, h_orig, w_orig), device=device)
    weight_mask = get_hanning_mask(patch_size).to(device)
    weight_mask = weight_mask.unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1)
    
    pos_coeff = Posterior_Coefficients(args, device)

    steps_h = list(range(0, h_orig - patch_size + 1, stride))
    if steps_h[-1] != h_orig - patch_size: steps_h.append(h_orig - patch_size)
    
    steps_w = list(range(0, w_orig - patch_size + 1, stride))
    if steps_w[-1] != w_orig - patch_size: steps_w.append(w_orig - patch_size)

    total_patches = len(steps_h) * len(steps_w)
    processed = 0

    print(f"🧩 图像尺寸: {w_orig}x{h_orig}, Stride: {stride}, Total Patches: {total_patches}")
    
    for y in steps_h:
        for x in steps_w:
            processed += 1
            print(f"\r  ⚡ 生成进度: {processed}/{total_patches} ...", end="")
            
            source_patch = img_tensor[:, :, y:y+patch_size, x:x+patch_size]
            noise_patch = torch.randn_like(source_patch)
            x_init = torch.cat((noise_patch, source_patch), axis=1)
            
            predicted_patch = sample_from_model(pos_coeff, netG, args.num_timesteps, x_init, args)
            
            output_canvas[:, :, y:y+patch_size, x:x+patch_size] += predicted_patch * weight_mask
            count_map[:, :, y:y+patch_size, x:x+patch_size] += weight_mask

    print("\n✅ 融合完成。")
    
    # 4. 生成最终结果
    final_output = output_canvas / (count_map + 1e-8)
    final_img = final_output.squeeze().permute(1, 2, 0).cpu().numpy()
    fake_final = normalize_to_uint8(final_img)

    # 5. 计算指标 (包含 LPIPS)
    val_psnr = 0.0
    val_ssim = 0.0
    val_lpips = 0.0
    
    if target_rgb is not None:
        # A. 计算 PSNR
        val_psnr = psnr(target_rgb, fake_final, data_range=255)
        
        # B. 计算 SSIM
        try:
            val_ssim = ssim(target_rgb, fake_final, win_size=3, data_range=255, channel_axis=2)
        except:
            val_ssim = ssim(target_rgb, fake_final, win_size=3, data_range=255, multichannel=True)
            
        # C. 计算 LPIPS 【新增核心代码】
        try:
            # 初始化 LPIPS (使用 AlexNet, 它是标准)
            loss_fn_lpips = lpips.LPIPS(net='alex').to(device)
            
            # 数据准备：需要 [B, 3, H, W] 且范围在 [-1, 1]
            # 1. 处理 Ground Truth
            target_tensor = torch.from_numpy(target_rgb).permute(2, 0, 1).unsqueeze(0).float().to(device)
            target_tensor = target_tensor / 127.5 - 1.0 # [0,255] -> [-1,1]
            
            # 2. 处理 Fake (使用 fake_final 转换，确保所见即所测)
            fake_tensor = torch.from_numpy(fake_final).permute(2, 0, 1).unsqueeze(0).float().to(device)
            fake_tensor = fake_tensor / 127.5 - 1.0 # [0,255] -> [-1,1]
            
            with torch.no_grad():
                val_lpips = loss_fn_lpips(fake_tensor, target_tensor).item()
                
            # 清理显存
            del loss_fn_lpips
            torch.cuda.empty_cache()
            
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
        cv2.putText(img, text, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
        return img

    show_fundus = put_text(img_rgb, "Input (FS)")
    
    text_fake = f"Ours"
    if target_rgb is not None:
        # 在图片上显示三个指标
        text_fake = f"P:{val_psnr:.2f} S:{val_ssim:.3f} L:{val_lpips:.3f}"
        
    show_fake = put_text(fake_final, text_fake, color=(0, 255, 255)) # 黄色字显示生成的

    img_list = [show_fundus, show_fake]
    
    if target_rgb is not None:
        show_real = put_text(target_rgb, "Ground Truth")
        img_list.append(show_real)

    comparison = np.hstack(img_list)
    
    filename = os.path.basename(fundus_path)
    save_path = os.path.join(args.output_path, f"RESULT_{filename}")
    
    cv2.imwrite(save_path, cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR))
    print(f"💾 结果已保存: {save_path}")

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
    
    # NCSNpp 参数
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
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--input_image', type=str, required=True)
    parser.add_argument('--target_image', type=str, default=None)
    parser.add_argument('--output_path', type=str, default='./inference_results')
    parser.add_argument('--gpu', type=int, default=0)

    args = parser.parse_args()

    device = torch.device(f'cuda:{args.gpu}')
    
    print(f"📂 加载模型权重: {args.model_path}")
    netG = NCSNpp(args).to(device)
    
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
    
    netG.load_state_dict(state_dict)
    netG.eval()

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    predict_and_stitch(netG, args.input_image, args.target_image, args, device)

if __name__ == '__main__':
    main()