import argparse
import torch
import numpy as np
import scipy.io as sio 
import h5py
import os
import torchvision
from backbones.ncsnpp_generator_adagn import NCSNpp 
from dataset import CreateDatasetSynthesis 
import torchvision.transforms as transforms

# ==========================================
# 辅助函数区域
# ==========================================
def psnr(img1, img2):
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    return 20 * torch.log10(img1.max() / torch.sqrt(mse))

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

def get_time_schedule(args, device):
    n_timestep = args.num_timesteps
    eps_small = 1e-3
    t = np.arange(0, n_timestep + 1, dtype=np.float64)
    t = t / n_timestep
    t = torch.from_numpy(t) * (1. - eps_small)  + eps_small
    return t.to(device)

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

def sample_posterior(coefficients, x_0,x_t, t):
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

def sample_from_model(coefficients, generator, n_time, x_init, T, opt):
    C = opt.num_channels
    x = x_init[:, 0:C, :, :]      
    source = x_init[:, C:, :, :]  
    
    with torch.no_grad():
        for i in reversed(range(n_time)):
            t = torch.full((x.size(0),), i, dtype=torch.int64).to(x.device)
            t_time = t
            latent_z = torch.randn(x.size(0), opt.nz, device=x.device)
            x_0 = generator(torch.cat((x,source),axis=1), t_time, latent_z)
            x_new = sample_posterior(coefficients, x_0[:,0:C,:], x, t)
            x = x_new.detach() 
    return x 

def load_checkpoint(checkpoint_dir, netG, name_of_network, epoch, device='cuda:0'):
    checkpoint_file = checkpoint_dir.format(name_of_network, epoch)  
    print(f"Loading checkpoint: {checkpoint_file}")
    checkpoint = torch.load(checkpoint_file, map_location=device)
    ckpt = checkpoint
    if 'module.' in list(ckpt.keys())[0]:
        new_ckpt = {}
        for key in list(ckpt.keys()):
            new_ckpt[key[7:]] = ckpt.pop(key)
        ckpt = new_ckpt
    netG.load_state_dict(ckpt)
    netG.eval() 

# ==========================================
# 主测试逻辑
# ==========================================
def sample_and_test(args):
    torch.manual_seed(args.seed)
    torch.cuda.set_device(args.gpu_chose)
    device = torch.device('cuda:{}'.format(args.gpu_chose))
    epoch_chosen = args.which_epoch 
    
    to_range_0_1 = lambda x: (x + 1.) / 2.

    # 1. 加载数据集 
    # 这里读取 .mat 文件，里面的数据已经是 256x256 的了 (由 preprocess_final.py 生成)
    dataset = CreateDatasetSynthesis('test', args.input_path, args.contrast1, args.contrast2)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)
    
    # 2. 初始化模型
    gen_diffusive_1 = NCSNpp(args).to(device)
    gen_diffusive_2 = NCSNpp(args).to(device)

    exp_path = os.path.join(args.output_path, args.exp)
    checkpoint_file = exp_path + "/{}_{}.pth"
    
    print(f"--- 正在加载 Epoch {epoch_chosen} 的模型 ---")
    load_checkpoint(checkpoint_file, gen_diffusive_1, 'gen_diffusive_1', str(epoch_chosen), device=device)
    load_checkpoint(checkpoint_file, gen_diffusive_2, 'gen_diffusive_2', str(epoch_chosen), device=device)

    T = get_time_schedule(args, device)
    pos_coeff = Posterior_Coefficients(args, device)
          
    save_dir = exp_path + "/generated_samples/epoch_{}".format(epoch_chosen)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    loss1 = np.zeros((1, len(data_loader)))
    loss2 = np.zeros((1, len(data_loader)))
    
    # 使用 args.image_size (默认256) 初始化数组
    img_h = args.image_size
    img_w = args.image_size
    syn_im1 = np.zeros((len(data_loader), args.num_channels, img_h, img_w))
    syn_im2 = np.zeros((len(data_loader), args.num_channels, img_h, img_w))
    
    # -------------------------------------------------------
    # 任务 1: Contrast 2 -> Contrast 1 (例如 FFA -> Fundus)
    # -------------------------------------------------------
    print(f"Generating {args.contrast1} from {args.contrast2}...")
    for iteration, (x , y) in enumerate(data_loader): 
        # x 是 contrast1 (Fundus), y 是 contrast2 (FFA)
        real_data = x.to(device, non_blocking=True)   # Fundus (Target)
        source_data = y.to(device, non_blocking=True) # FFA (Source)
        
        # 拼接噪声和源图像作为输入
        x1_t = torch.cat((torch.randn_like(real_data), source_data), axis=1)
        fake_sample1 = sample_from_model(pos_coeff, gen_diffusive_1, args.num_timesteps, x1_t, T, args)
    
        # 归一化处理
        fake_sample1 = to_range_0_1(fake_sample1); fake_sample1 = fake_sample1 / fake_sample1.max()
        real_data = to_range_0_1(real_data); real_data = real_data / real_data.max()
        source_data = to_range_0_1(source_data); source_data = source_data / source_data.max() 
        
        # 保存数据用于 .mat
        syn_im1[iteration, :, :, :] = fake_sample1.cpu().numpy()
        
        # 计算 PSNR
        loss1[0, iteration] = psnr(fake_sample1, real_data).cpu().numpy()
        if iteration % 10 == 0:
            print(f"Batch {iteration} PSNR: {loss1[0, iteration]:.2f}")
        
        # 保存图片 (Source | Generated | Ground Truth)
        # 注意：这里直接拼接，不做任何 crop
        combined_img = torch.cat((source_data, fake_sample1, real_data), axis=-1)
        torchvision.utils.save_image(combined_img, '{}/{}_samples1_{}.jpg'.format(save_dir, 'test', iteration), normalize=True)

    # -------------------------------------------------------
    # 任务 2: Contrast 1 -> Contrast 2 (例如 Fundus -> FFA)
    # -------------------------------------------------------
    print(f"Generating {args.contrast2} from {args.contrast1}...")
    for iteration, (x , y) in enumerate(data_loader): 
        real_data = y.to(device, non_blocking=True)   # FFA (Target)
        source_data = x.to(device, non_blocking=True) # Fundus (Source)
        
        x2_t = torch.cat((torch.randn_like(real_data), source_data), axis=1)
        fake_sample2 = sample_from_model(pos_coeff, gen_diffusive_2, args.num_timesteps, x2_t, T, args)
    
        fake_sample2 = to_range_0_1(fake_sample2); fake_sample2 = fake_sample2 / fake_sample2.max()
        real_data = to_range_0_1(real_data); real_data = real_data / real_data.max()
        source_data = to_range_0_1(source_data); source_data = source_data / source_data.max() 
        
        syn_im2[iteration, :, :, :] = fake_sample2.cpu().numpy()
        
        loss2[0, iteration] = psnr(fake_sample2, real_data).cpu().numpy()
        
        combined_img = torch.cat((source_data, fake_sample2, real_data), axis=-1)
        torchvision.utils.save_image(combined_img, '{}/{}_samples2_{}.jpg'.format(save_dir, 'test', iteration), normalize=True)

    # 3. 保存最终结果
    print("Avg PSNR T2->T1:", np.nanmean(loss1))
    np.save('{}/psnr_values1.npy'.format(save_dir), loss1)

    print("Avg PSNR T1->T2:", np.nanmean(loss2))
    np.save('{}/psnr_values2.npy'.format(save_dir), loss2)

    try:
        f = h5py.File(save_dir + '/im_syn.mat',  "w")
        f.create_dataset('images_'+args.contrast1+'syn', data=syn_im1)
        f.create_dataset('images_'+args.contrast2+'syn', data=syn_im2)
        f.close()
        print("Mat file saved successfully.")
    except Exception as e:
        print(f"h5py error: {e}, using scipy...")
        sio.savemat(save_dir + '/im_syn.mat', {
            'images_'+args.contrast1+'syn': syn_im1,
            'images_'+args.contrast2+'syn': syn_im2
        })

if __name__ == '__main__':
    parser = argparse.ArgumentParser('syndiff parameters')
    parser.add_argument('--seed', type=int, default=1024)
    parser.add_argument('--compute_fid', action='store_true', default=False)
    parser.add_argument('--epoch_id', type=int, default=1000)
    parser.add_argument('--num_channels', type=int, default=3)
    parser.add_argument('--centered', action='store_false', default=True)
    parser.add_argument('--use_geometric', action='store_true', default=False)
    parser.add_argument('--beta_min', type=float, default=0.1)
    parser.add_argument('--beta_max', type=float, default=20.)
    
    parser.add_argument('--num_channels_dae', type=int, default=64)
    parser.add_argument('--n_mlp', type=int, default=3)
    parser.add_argument('--ch_mult', nargs='+', type=int)
    parser.add_argument('--num_res_blocks', type=int, default=2)
    parser.add_argument('--attn_resolutions', default=(16,))
    parser.add_argument('--dropout', type=float, default=0.)
    parser.add_argument('--resamp_with_conv', action='store_false', default=True)
    parser.add_argument('--conditional', action='store_false', default=True)
    parser.add_argument('--fir', action='store_false', default=True)
    parser.add_argument('--fir_kernel', default=[1, 3, 3, 1])
    parser.add_argument('--skip_rescale', action='store_false', default=True)
    parser.add_argument('--resblock_type', default='biggan')
    parser.add_argument('--progressive', type=str, default='none', choices=['none', 'output_skip', 'residual'])
    parser.add_argument('--progressive_input', type=str, default='residual', choices=['none', 'input_skip', 'residual'])
    parser.add_argument('--progressive_combine', type=str, default='sum', choices=['sum', 'cat'])
    parser.add_argument('--embedding_type', type=str, default='positional', choices=['positional', 'fourier'])
    parser.add_argument('--fourier_scale', type=float, default=16.)
    parser.add_argument('--not_use_tanh', action='store_true', default=False)
    
    parser.add_argument('--exp', default='ixi_synth', help='name of experiment')
    parser.add_argument('--input_path', help='path to input data')
    parser.add_argument('--output_path', help='path to output saves')
    
    parser.add_argument('--dataset', default='cifar10')
    parser.add_argument('--image_size', type=int, default=256, help='size of image') # 确保这里是 256
    
    parser.add_argument('--nz', type=int, default=100)
    parser.add_argument('--num_timesteps', type=int, default=4)
    
    parser.add_argument('--z_emb_dim', type=int, default=256)
    parser.add_argument('--t_emb_dim', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--lr_g', type=float, default=1.5e-4)
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--beta2', type=float, default=0.9)
    
    parser.add_argument('--contrast1', type=str, default='T1')
    parser.add_argument('--contrast2', type=str, default='T2')
    parser.add_argument('--which_epoch', type=int, default=50)
    parser.add_argument('--gpu_chose', type=int, default=0)
    parser.add_argument('--source', type=str, default='T2')   
    
    args = parser.parse_args()
    
    sample_and_test(args)