import argparse
import torch
import numpy as np
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from dataset import CreateDatasetSynthesis
from torch.multiprocessing import Process
import torch.distributed as dist
import shutil
from skimage.metrics import peak_signal_noise_ratio as psnr

from tqdm import tqdm
from utils_train import get_logger, LossTracker

# 分布式训练场景下的深度学习模型基础框架
def copy_source(file, output_dir):
    shutil.copyfile(file, os.path.join(output_dir, os.path.basename(file)))

def broadcast_params(params):
    for param in params:
        dist.broadcast(param.data, src=0)

# %% Diffusion coefficients 
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

class Diffusion_Coefficients():
    def __init__(self, args, device):
        self.sigmas, self.a_s, _ = get_sigma_schedule(args, device=device)
        self.a_s_cum = np.cumprod(self.a_s.cpu())
        self.sigmas_cum = np.sqrt(1 - self.a_s_cum ** 2)
        self.a_s_prev = self.a_s.clone()
        self.a_s_prev[-1] = 1
        
        self.a_s_cum = self.a_s_cum.to(device)
        self.sigmas_cum = self.sigmas_cum.to(device)
        self.a_s_prev = self.a_s_prev.to(device)
    
def q_sample(coeff, x_start, t, *, noise=None):
    if noise is None:
      noise = torch.randn_like(x_start)
      
    x_t = extract(coeff.a_s_cum, t, x_start.shape) * x_start + \
          extract(coeff.sigmas_cum, t, x_start.shape) * noise
    
    return x_t

def q_sample_pairs(coeff, x_start, t):
    noise = torch.randn_like(x_start)
    x_t = q_sample(coeff, x_start, t)
    x_t_plus_one = extract(coeff.a_s, t+1, x_start.shape) * x_t + \
                   extract(coeff.sigmas, t+1, x_start.shape) * noise
    
    return x_t, x_t_plus_one

# %% posterior sampling
class Posterior_Coefficients():
    def __init__(self, args, device):
        
        _, _, self.betas = get_sigma_schedule(args, device=device)
        
        #we don't need the zeros
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
            x_new = sample_posterior(coefficients, x_0[:,0:C,:,:], x, t)
            x = x_new.detach()
        
    return x
# %%
def train_syndiff(rank, gpu, args):
    
    from backbones.discriminator import Discriminator_small, Discriminator_large
    from backbones.ncsnpp_generator_adagn import NCSNpp
    import backbones.generator_resnet
    from utils.EMA import EMA
    
    torch.manual_seed(args.seed + rank)
    torch.cuda.manual_seed(args.seed + rank)
    torch.cuda.manual_seed_all(args.seed + rank)
    device = torch.device('cuda:{}'.format(gpu))
    
    batch_size = args.batch_size
    nz = args.nz 
    
    # ==========================================
    # ✅【新增】初始化日志工具 (只在 Rank 0 执行)
    # ==========================================
    exp_path = os.path.join(args.output_path, args.exp)
    if rank == 0:
        # 使用工具类自动创建带时间戳的目录，或者沿用你原有的 exp_path
        # 这里为了兼容你的代码逻辑，我们直接用你原来的 exp_path，但加上日志功能
        if not os.path.exists(exp_path):
            os.makedirs(exp_path)
        
        # 初始化 logger 和 tracker
        logger = get_logger(exp_path)
        tracker = LossTracker(exp_path)
        
        logger.info(f"Start Training on Rank {rank}")
        
        # 复制代码备份
        copy_source(__file__, exp_path)
        if not os.path.exists(os.path.join(exp_path, 'backbones')):
            shutil.copytree('./backbones', os.path.join(exp_path, 'backbones'))
    
    dataset = CreateDatasetSynthesis(phase = "train", input_path = args.input_path, contrast1 = args.contrast1, contrast2 = args.contrast2)
    dataset_val = CreateDatasetSynthesis(phase = "val", input_path = args.input_path, contrast1 = args.contrast1, contrast2 = args.contrast2 )

    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=args.world_size, rank=rank)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, sampler=train_sampler, drop_last = True)
    
    val_sampler = torch.utils.data.distributed.DistributedSampler(dataset_val, num_replicas=args.world_size, rank=rank)
    data_loader_val = torch.utils.data.DataLoader(dataset_val, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, sampler=val_sampler, drop_last = True)

    val_l1_loss=np.zeros([2,args.num_epoch,len(data_loader_val)])
    val_psnr_values=np.zeros([2,args.num_epoch,len(data_loader_val)])
    
    if rank == 0:
        logger.info('train data size:'+str(len(data_loader)))
        logger.info('val data size:'+str(len(data_loader_val)))
        
    to_range_0_1 = lambda x: (x + 1.) / 2.
    
    # Loss 实例化
    # ✅ 修正代码

    # 模型实例化
    gen_diffusive_1 = NCSNpp(args).to(device)
    gen_diffusive_2 = NCSNpp(args).to(device) 
    gen_non_diffusive_1to2 = backbones.generator_resnet.define_G(netG='resnet_6blocks',gpu_ids=[gpu], input_nc=args.num_channels, output_nc=args.num_channels)
    gen_non_diffusive_2to1 = backbones.generator_resnet.define_G(netG='resnet_6blocks',gpu_ids=[gpu], input_nc=args.num_channels, output_nc=args.num_channels)

    disc_in_channels = 2 * args.num_channels
    disc_diffusive_1 = Discriminator_large(nc = disc_in_channels, ngf = args.ngf, t_emb_dim = args.t_emb_dim, act=nn.LeakyReLU(0.2)).to(device)
    disc_diffusive_2 = Discriminator_large(nc = disc_in_channels, ngf = args.ngf, t_emb_dim = args.t_emb_dim, act=nn.LeakyReLU(0.2)).to(device)
    disc_non_diffusive_cycle1 = backbones.generator_resnet.define_D(gpu_ids=[gpu], input_nc=args.num_channels)
    disc_non_diffusive_cycle2 = backbones.generator_resnet.define_D(gpu_ids=[gpu], input_nc=args.num_channels)
    
    # 广播参数
    broadcast_params(gen_diffusive_1.parameters())
    broadcast_params(gen_diffusive_2.parameters())
    broadcast_params(gen_non_diffusive_1to2.parameters())
    broadcast_params(gen_non_diffusive_2to1.parameters())
    broadcast_params(disc_diffusive_1.parameters())
    broadcast_params(disc_diffusive_2.parameters())
    broadcast_params(disc_non_diffusive_cycle1.parameters())
    broadcast_params(disc_non_diffusive_cycle2.parameters())
    
    # 优化器
    optimizer_disc_diffusive_1 = optim.Adam(disc_diffusive_1.parameters(), lr=args.lr_d, betas = (args.beta1, args.beta2))
    optimizer_disc_diffusive_2 = optim.Adam(disc_diffusive_2.parameters(), lr=args.lr_d, betas = (args.beta1, args.beta2))
    optimizer_gen_diffusive_1 = optim.Adam(gen_diffusive_1.parameters(), lr=args.lr_g, betas = (args.beta1, args.beta2))
    optimizer_gen_diffusive_2 = optim.Adam(gen_diffusive_2.parameters(), lr=args.lr_g, betas = (args.beta1, args.beta2))
    optimizer_gen_non_diffusive_1to2 = optim.Adam(gen_non_diffusive_1to2.parameters(), lr=args.lr_g, betas = (args.beta1, args.beta2))
    optimizer_gen_non_diffusive_2to1 = optim.Adam(gen_non_diffusive_2to1.parameters(), lr=args.lr_g, betas = (args.beta1, args.beta2))
    optimizer_disc_non_diffusive_cycle1 = optim.Adam(disc_non_diffusive_cycle1.parameters(), lr=args.lr_d, betas = (args.beta1, args.beta2))
    optimizer_disc_non_diffusive_cycle2 = optim.Adam(disc_non_diffusive_cycle2.parameters(), lr=args.lr_d, betas = (args.beta1, args.beta2))    
    
    if args.use_ema:
        optimizer_gen_diffusive_1 = EMA(optimizer_gen_diffusive_1, ema_decay=args.ema_decay)
        optimizer_gen_diffusive_2 = EMA(optimizer_gen_diffusive_2, ema_decay=args.ema_decay)
        optimizer_gen_non_diffusive_1to2 = EMA(optimizer_gen_non_diffusive_1to2, ema_decay=args.ema_decay)
        optimizer_gen_non_diffusive_2to1 = EMA(optimizer_gen_non_diffusive_2to1, ema_decay=args.ema_decay)
        
    # 调度器
    scheduler_gen_diffusive_1 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_gen_diffusive_1, args.num_epoch, eta_min=1e-5)
    scheduler_gen_diffusive_2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_gen_diffusive_2, args.num_epoch, eta_min=1e-5)
    scheduler_gen_non_diffusive_1to2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_gen_non_diffusive_1to2, args.num_epoch, eta_min=1e-5)
    scheduler_gen_non_diffusive_2to1 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_gen_non_diffusive_2to1, args.num_epoch, eta_min=1e-5)    
    scheduler_disc_diffusive_1 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_disc_diffusive_1, args.num_epoch, eta_min=1e-5)
    scheduler_disc_diffusive_2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_disc_diffusive_2, args.num_epoch, eta_min=1e-5)
    scheduler_disc_non_diffusive_cycle1 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_disc_non_diffusive_cycle1, args.num_epoch, eta_min=1e-5)
    scheduler_disc_non_diffusive_cycle2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_disc_non_diffusive_cycle2, args.num_epoch, eta_min=1e-5)
    
    # DDP Wrappers
    gen_diffusive_1 = nn.parallel.DistributedDataParallel(gen_diffusive_1, device_ids=[gpu])
    gen_diffusive_2 = nn.parallel.DistributedDataParallel(gen_diffusive_2, device_ids=[gpu])
    gen_non_diffusive_1to2 = nn.parallel.DistributedDataParallel(gen_non_diffusive_1to2, device_ids=[gpu])
    gen_non_diffusive_2to1 = nn.parallel.DistributedDataParallel(gen_non_diffusive_2to1, device_ids=[gpu])    
    disc_diffusive_1 = nn.parallel.DistributedDataParallel(disc_diffusive_1, device_ids=[gpu])
    disc_diffusive_2 = nn.parallel.DistributedDataParallel(disc_diffusive_2, device_ids=[gpu])
    disc_non_diffusive_cycle1 = nn.parallel.DistributedDataParallel(disc_non_diffusive_cycle1, device_ids=[gpu])
    disc_non_diffusive_cycle2 = nn.parallel.DistributedDataParallel(disc_non_diffusive_cycle2, device_ids=[gpu])
    
    coeff = Diffusion_Coefficients(args, device)
    pos_coeff = Posterior_Coefficients(args, device)
    T = get_time_schedule(args, device)
    
    if args.resume:
        checkpoint_file = os.path.join(exp_path, 'content.pth')
        checkpoint = torch.load(checkpoint_file, map_location=device)
        init_epoch = checkpoint['epoch']
        epoch = init_epoch
        gen_diffusive_1.load_state_dict(checkpoint['gen_diffusive_1_dict'])
        gen_diffusive_2.load_state_dict(checkpoint['gen_diffusive_2_dict'])
        gen_non_diffusive_1to2.load_state_dict(checkpoint['gen_non_diffusive_1to2_dict'])
        gen_non_diffusive_2to1.load_state_dict(checkpoint['gen_non_diffusive_2to1_dict'])        
        
        optimizer_gen_diffusive_1.load_state_dict(checkpoint['optimizer_gen_diffusive_1'])
        scheduler_gen_diffusive_1.load_state_dict(checkpoint['scheduler_gen_diffusive_1'])
        optimizer_gen_diffusive_2.load_state_dict(checkpoint['optimizer_gen_diffusive_2'])
        scheduler_gen_diffusive_2.load_state_dict(checkpoint['scheduler_gen_diffusive_2']) 
        optimizer_gen_non_diffusive_1to2.load_state_dict(checkpoint['optimizer_gen_non_diffusive_1to2'])
        scheduler_gen_non_diffusive_1to2.load_state_dict(checkpoint['scheduler_gen_non_diffusive_1to2'])
        optimizer_gen_non_diffusive_2to1.load_state_dict(checkpoint['optimizer_gen_non_diffusive_2to1'])
        scheduler_gen_non_diffusive_2to1.load_state_dict(checkpoint['scheduler_gen_non_diffusive_2to1'])          
        
        disc_diffusive_1.load_state_dict(checkpoint['disc_diffusive_1_dict'])
        optimizer_disc_diffusive_1.load_state_dict(checkpoint['optimizer_disc_diffusive_1'])
        scheduler_disc_diffusive_1.load_state_dict(checkpoint['scheduler_disc_diffusive_1'])
        disc_diffusive_2.load_state_dict(checkpoint['disc_diffusive_2_dict'])
        optimizer_disc_diffusive_2.load_state_dict(checkpoint['optimizer_disc_diffusive_2'])
        scheduler_disc_diffusive_2.load_state_dict(checkpoint['scheduler_disc_diffusive_2'])    
        
        disc_non_diffusive_cycle1.load_state_dict(checkpoint['disc_non_diffusive_cycle1_dict'])
        optimizer_disc_non_diffusive_cycle1.load_state_dict(checkpoint['optimizer_disc_non_diffusive_cycle1'])
        scheduler_disc_non_diffusive_cycle1.load_state_dict(checkpoint['scheduler_disc_non_diffusive_cycle1'])
        disc_non_diffusive_cycle2.load_state_dict(checkpoint['disc_non_diffusive_cycle2_dict'])
        optimizer_disc_non_diffusive_cycle2.load_state_dict(checkpoint['optimizer_disc_non_diffusive_cycle2'])
        scheduler_disc_non_diffusive_cycle2.load_state_dict(checkpoint['scheduler_disc_non_diffusive_cycle2'])
        global_step = checkpoint['global_step']
        if rank == 0: logger.info("=> loaded checkpoint (epoch {})".format(checkpoint['epoch']))
    else:
        global_step, epoch, init_epoch = 0, 0, 0
        
    # ==========================================================
    # ✅ 【第二步：新增】初始化 VGG-19 感知损失提取器
    # ==========================================================
    # 因为你用了 DDP 多卡训练 (有 rank 变量)，为了防止报错，
    # 最稳妥的获取当前显卡 device 的写法如下：
    # ✅ 修正代码 (直接用上面已定义的 device)
    # 实例化，后续可以可以加到第四层
    # ==========================================================
        
    for epoch in range(init_epoch, args.num_epoch+1):
        train_sampler.set_epoch(epoch)
        
        # 👇 【新增】用于记录当前 Epoch 累加值的字典
        epoch_losses_raw = {"G_adv": 0.0, "G_cycle": 0.0, "G_l1": 0.0}
        
        # ✅【新增】进度条封装 (仅在 Rank 0 显示)
        # 这样你就不会看到满屏滚动的 print，而是一个优雅的进度条
        if rank == 0:
            loader_iter = tqdm(enumerate(data_loader), total=len(data_loader), desc=f"Epoch {epoch}/{args.num_epoch}", unit="img")
        else:
            loader_iter = enumerate(data_loader)
        
        # 用于记录 epoch 平均 loss
        epoch_g_loss = 0.0
        epoch_d_loss = 0.0
        
        for iteration, (x1, x2) in loader_iter:
            # 开启梯度
            for p in disc_diffusive_1.parameters(): p.requires_grad = True  
            for p in disc_diffusive_2.parameters(): p.requires_grad = True
            for p in disc_non_diffusive_cycle1.parameters(): p.requires_grad = True  
            for p in disc_non_diffusive_cycle2.parameters(): p.requires_grad = True          
            
            disc_diffusive_1.zero_grad()
            disc_diffusive_2.zero_grad()
            
            # --- 判别器训练开始 ---
            real_data1 = x1.to(device, non_blocking=True)
            real_data2 = x2.to(device, non_blocking=True)
            if real_data1.shape[1] > 3: real_data1 = real_data1[:, :3, :, :]
            if real_data2.shape[1] > 3: real_data2 = real_data2[:, :3, :, :]
            
            t1 = torch.randint(0, args.num_timesteps, (real_data1.size(0),), device=device)
            t2 = torch.randint(0, args.num_timesteps, (real_data2.size(0),), device=device)
            
            x1_t, x1_tp1 = q_sample_pairs(coeff, real_data1, t1)
            x1_t.requires_grad = True
            x2_t, x2_tp1 = q_sample_pairs(coeff, real_data2, t2)
            x2_t.requires_grad = True              
            
            D1_real = disc_diffusive_1(x1_t, t1, x1_tp1.detach()).view(-1)
            D2_real = disc_diffusive_2(x2_t, t2, x2_tp1.detach()).view(-1)   
            
            errD1_real = F.softplus(-D1_real).mean()            
            errD2_real = F.softplus(-D2_real).mean()   
            errD_real = errD1_real + errD2_real
            errD_real.backward(retain_graph=True)
            
            # 梯度惩罚 (Regularization)
            if args.lazy_reg is None:
                grad1_real = torch.autograd.grad(outputs=D1_real.sum(), inputs=x1_t, create_graph=True)[0]
                grad1_penalty = (grad1_real.view(grad1_real.size(0), -1).norm(2, dim=1) ** 2).mean()
                grad2_real = torch.autograd.grad(outputs=D2_real.sum(), inputs=x2_t, create_graph=True)[0]
                grad2_penalty = (grad2_real.view(grad2_real.size(0), -1).norm(2, dim=1) ** 2).mean()                
                grad_penalty = args.r1_gamma / 2 * grad1_penalty + args.r1_gamma / 2 * grad2_penalty
                grad_penalty.backward()
            else:
                if global_step % args.lazy_reg == 0:
                    grad1_real = torch.autograd.grad(outputs=D1_real.sum(), inputs=x1_t, create_graph=True)[0]
                    grad1_penalty = (grad1_real.view(grad1_real.size(0), -1).norm(2, dim=1) ** 2).mean()
                    grad2_real = torch.autograd.grad(outputs=D2_real.sum(), inputs=x2_t, create_graph=True)[0]
                    grad2_penalty = (grad2_real.view(grad2_real.size(0), -1).norm(2, dim=1) ** 2).mean()                
                    grad_penalty = args.r1_gamma / 2 * grad1_penalty + args.r1_gamma / 2 * grad2_penalty
                    grad_penalty.backward()
            
            latent_z1 = torch.randn(batch_size, nz, device=device) 
            latent_z2 = torch.randn(batch_size, nz, device=device) 

            x1_0_predict = gen_non_diffusive_2to1(real_data2) 
            x2_0_predict = gen_non_diffusive_1to2(real_data1)          

            x1_0_predict_diff = gen_diffusive_1(torch.cat((x1_tp1.detach(), x2_0_predict), axis=1), t1, latent_z1)
            x2_0_predict_diff = gen_diffusive_2(torch.cat((x2_tp1.detach(), x1_0_predict), axis=1), t2, latent_z2)

            C = args.num_channels
            x1_pos_sample = sample_posterior(pos_coeff, x1_0_predict_diff[:,0:C,:,:], x1_tp1, t1)
            x2_pos_sample = sample_posterior(pos_coeff, x2_0_predict_diff[:,0:C,:,:], x2_tp1, t2)

            output1 = disc_diffusive_1(x1_pos_sample, t1, x1_tp1.detach()).view(-1)
            output2 = disc_diffusive_2(x2_pos_sample, t2, x2_tp1.detach()).view(-1)       

            errD1_fake = F.softplus(output1).mean()
            errD2_fake = F.softplus(output2).mean()
            errD_fake = errD1_fake + errD2_fake
            errD_fake.backward()    

            errD = errD_real + errD_fake
            optimizer_disc_diffusive_1.step()
            optimizer_disc_diffusive_2.step()  

            disc_non_diffusive_cycle1.zero_grad()
            disc_non_diffusive_cycle2.zero_grad()

            D_cycle1_real = disc_non_diffusive_cycle1(real_data1).view(-1)
            D_cycle2_real = disc_non_diffusive_cycle2(real_data2).view(-1) 
            errD_cycle1_real = F.softplus(-D_cycle1_real).mean()            
            errD_cycle2_real = F.softplus(-D_cycle2_real).mean()   
            errD_cycle_real = errD_cycle1_real + errD_cycle2_real
            errD_cycle_real.backward(retain_graph=True) 

            x1_0_predict = gen_non_diffusive_2to1(real_data2)
            x2_0_predict = gen_non_diffusive_1to2(real_data1)

            D_cycle1_fake = disc_non_diffusive_cycle1(x1_0_predict).view(-1)
            D_cycle2_fake = disc_non_diffusive_cycle2(x2_0_predict).view(-1) 
            errD_cycle1_fake = F.softplus(D_cycle1_fake).mean()            
            errD_cycle2_fake = F.softplus(D_cycle2_fake).mean()   
            errD_cycle_fake = errD_cycle1_fake + errD_cycle2_fake
            errD_cycle_fake.backward()

            errD_cycle = errD_cycle_real + errD_cycle_fake
            optimizer_disc_non_diffusive_cycle1.step()
            optimizer_disc_non_diffusive_cycle2.step() 
            
            # --- 生成器训练开始 ---
            for p in disc_diffusive_1.parameters(): p.requires_grad = False
            for p in disc_diffusive_2.parameters(): p.requires_grad = False
            for p in disc_non_diffusive_cycle1.parameters(): p.requires_grad = False
            for p in disc_non_diffusive_cycle2.parameters(): p.requires_grad = False                

            gen_diffusive_1.zero_grad()
            gen_diffusive_2.zero_grad()
            gen_non_diffusive_1to2.zero_grad()
            gen_non_diffusive_2to1.zero_grad()   

            t1 = torch.randint(0, args.num_timesteps, (real_data1.size(0),), device=device)
            t2 = torch.randint(0, args.num_timesteps, (real_data2.size(0),), device=device)
            x1_t, x1_tp1 = q_sample_pairs(coeff, real_data1, t1)   
            x2_t, x2_tp1 = q_sample_pairs(coeff, real_data2, t2)            

            latent_z1 = torch.randn(batch_size, nz, device=device)
            latent_z2 = torch.randn(batch_size, nz, device=device)

            x1_0_predict = gen_non_diffusive_2to1(real_data2)
            x2_0_predict = gen_non_diffusive_1to2(real_data1)
            x2_0_predict_cycle = gen_non_diffusive_1to2(x1_0_predict)
            x1_0_predict_cycle = gen_non_diffusive_2to1(x2_0_predict)

            x1_0_predict_diff = gen_diffusive_1(torch.cat((x1_tp1.detach(), x2_0_predict), axis=1), t1, latent_z1)
            x2_0_predict_diff = gen_diffusive_2(torch.cat((x2_tp1.detach(), x1_0_predict), axis=1), t2, latent_z2)            

            x1_pos_sample = sample_posterior(pos_coeff, x1_0_predict_diff[:,0:C,:,:], x1_tp1, t1)
            x2_pos_sample = sample_posterior(pos_coeff, x2_0_predict_diff[:,0:C,:,:], x2_tp1, t2)

            output1 = disc_diffusive_1(x1_pos_sample, t1, x1_tp1.detach()).view(-1)
            output2 = disc_diffusive_2(x2_pos_sample, t2, x2_tp1.detach()).view(-1)  

            errG1 = F.softplus(-output1).mean()
            errG2 = F.softplus(-output2).mean()
            errG_adv = errG1 + errG2

            D_cycle1_fake = disc_non_diffusive_cycle1(x1_0_predict).view(-1)
            D_cycle2_fake = disc_non_diffusive_cycle2(x2_0_predict).view(-1) 
            errG_cycle_adv1 = F.softplus(-D_cycle1_fake).mean()            
            errG_cycle_adv2 = F.softplus(-D_cycle2_fake).mean()   
            errG_cycle_adv = errG_cycle_adv1 + errG_cycle_adv2

                        # ===================== 1. 基础L1损失 =====================
            errG1_L1 = F.l1_loss(x1_0_predict_diff[:,0:C,:,:], real_data1)
            errG2_L1 = F.l1_loss(x2_0_predict_diff[:,0:C,:,:], real_data2)
            errG_L1 = errG1_L1 + errG2_L1 

            # ===================== 2. 循环一致性损失 =====================
            errG1_cycle = F.l1_loss(x1_0_predict_cycle, real_data1)
            errG2_cycle = F.l1_loss(x2_0_predict_cycle, real_data2)            
            errG_cycle = errG1_cycle + errG2_cycle  

            # ======================================
            # 绝对复刻原作者的权重环境 (基线控制变量)
            # ======================================
            lambda_l1_loss = 0.5  # 严格遵从原作者的扩散模型软约束设定

            # ===================== 🔥 终极总损失 =====================
            # 完全还原原始公式：0.5 * Cycle + Adv + Cycle_Adv + 0.5 * 跨域L1
            errG = lambda_l1_loss * errG_cycle \
                + errG_adv \
                + errG_cycle_adv \
                + lambda_l1_loss * errG_L1
            # ======================================================================

            errG.backward()
            
           # 👇 【新增】累加裸值 (Raw) —— 未加权的原始损失值
           # 👇 【修复版】累加裸值 (Raw) —— 自动初始化字典键，解决KeyError
            # 👇 【终极修复】先初始化所有字典键，再累加（永不报错）
            # ===================== Raw 损失裸值 初始化 + 累加 =====================
            epoch_losses_raw["G_adv"] += errG_adv.item()
            epoch_losses_raw["G_cycle"] += errG_cycle.item()
            epoch_losses_raw["G_l1"] += errG_L1.item()
            optimizer_gen_diffusive_1.step()
            optimizer_gen_diffusive_2.step()
            optimizer_gen_non_diffusive_1to2.step()
            optimizer_gen_non_diffusive_2to1.step()           

            global_step += 1

            # ✅【新增】更新进度条后缀 (实时查看 Loss)
            # ✅【新增】更新进度条后缀 (实时查看 Loss)
            if rank == 0:
                epoch_g_loss += errG.item()
                epoch_d_loss += errD.item()
                
                # 【关键修复】删掉所有 args.，用本地写死的权重变量！
                
                
                # 进度条显示（无任何args，纯本地变量）
                loader_iter.set_postfix({
                    "G": f"{errG.item():.3f}",
                    "D": f"{errD.item():.3f}",
                    "Cyc": f"{(lambda_l1_loss * errG_cycle).item():.3f}",
                    "L1": f"{(lambda_l1_loss * errG_L1).item():.3f}"
                })
        # 学习率衰减
        if not args.no_lr_decay:
            scheduler_gen_diffusive_1.step()
            scheduler_gen_diffusive_2.step()
            scheduler_gen_non_diffusive_1to2.step()
            scheduler_gen_non_diffusive_2to1.step()
            scheduler_disc_diffusive_1.step()
            scheduler_disc_diffusive_2.step()
            scheduler_disc_non_diffusive_cycle1.step()
            scheduler_disc_non_diffusive_cycle2.step()

        # ✅【新增】每个 Epoch 结束后更新 Loss 曲线图
        # ✅【新增】每个 Epoch 结束后更新 Loss 曲线图
        if rank == 0:
            avg_g_loss = epoch_g_loss / len(data_loader)
            avg_d_loss = epoch_d_loss / len(data_loader)
            tracker.update(epoch, avg_g_loss)
            logger.info(f"Epoch {epoch} Done. Avg G Loss: {avg_g_loss:.4f}, Avg D Loss: {avg_d_loss:.4f}")
            
            # 👇 【严格对齐缩进】计算并打印极其详尽的 Loss 拆解
            num_batches = len(data_loader)
            
            for k in epoch_losses_raw:
                epoch_losses_raw[k] /= num_batches
            tracker.update_named(epoch, {
                "G_adv": epoch_losses_raw["G_adv"],
                "G_cycle": epoch_losses_raw["G_cycle"],
                "G_l1": epoch_losses_raw["G_l1"],
                "D_total": avg_d_loss,
            })
                
            logger.info(f"--- Epoch {epoch} Loss Breakdown ---")
            logger.info(f"Raw Losses: { {k: round(v, 4) for k, v in epoch_losses_raw.items()} }")
            logger.info(f"G_l1: {epoch_losses_raw['G_l1']:.6f}")
            logger.info(f"G_cycle: {epoch_losses_raw['G_cycle']:.6f}")
            logger.info(f"G_adv: {epoch_losses_raw['G_adv']:.6f}")
            logger.info(f"D_total: {avg_d_loss:.6f}")

            def save_rgb(tensor, path):
                if tensor.shape[1] > 3:
                    torchvision.utils.save_image(tensor[:, :3, :, :], path, normalize=True)
                else:
                    torchvision.utils.save_image(tensor, path, normalize=True)

            if epoch % 10 == 0:
                save_rgb(x1_pos_sample, os.path.join(exp_path, 'xpos1_epoch_{}.png'.format(epoch)))
                save_rgb(x2_pos_sample, os.path.join(exp_path, 'xpos2_epoch_{}.png'.format(epoch)))
                
                # 2. 生成并保存 Domain 2 -> Domain 1 的样本
                # 拼接噪声和源域数据
                x1_t = torch.cat((torch.randn_like(real_data1), real_data2), axis=1)
                fake_sample1 = sample_from_model(pos_coeff, gen_diffusive_1, args.num_timesteps, x1_t, T, args)
                
                # 【核心修改点】拼接展示时只用RGB切片
                vis_real2 = real_data2[:, :3, :, :]
                vis_fake1 = fake_sample1[:, :3, :, :]
                fake_sample1_vis = torch.cat((vis_real2, vis_fake1), axis=-1)
                torchvision.utils.save_image(fake_sample1_vis, os.path.join(exp_path, 'sample1_discrete_epoch_{}.png'.format(epoch)), normalize=True)
                
                # 生成粗糙预测
                pred1 = gen_non_diffusive_2to1(real_data2)
                
                # 生成扩散精修结果
                x2_t = torch.cat((torch.randn_like(real_data2), pred1), axis=1)
                fake_sample2_tilda = gen_diffusive_2(x2_t , t2, latent_z2)   
                
                # 【核心修改点】复杂对比图的RGB切片
                vis_pred1 = pred1[:, :3, :, :]
                vis_cycle1 = gen_non_diffusive_1to2(pred1)[:, :3, :, :]
                vis_fake2_tilda = fake_sample2_tilda[:, :3, :, :]
                
                # 拼接：真实图 | 粗糙预测 | 循环重建 | 扩散精修
                pred1_vis = torch.cat((vis_real2, vis_pred1, vis_cycle1, vis_fake2_tilda), axis=-1)
                torchvision.utils.save_image(pred1_vis, os.path.join(exp_path, 'sample1_translated_epoch_{}.png'.format(epoch)), normalize=True)

                # 3. 生成并保存 Domain 1 -> Domain 2 的样本 (同上)
                x2_t = torch.cat((torch.randn_like(real_data2), real_data1), axis=1)
                fake_sample2 = sample_from_model(pos_coeff, gen_diffusive_2, args.num_timesteps, x2_t, T, args)
                
                vis_real1 = real_data1[:, :3, :, :]
                vis_fake2 = fake_sample2[:, :3, :, :]
                fake_sample2_vis = torch.cat((vis_real1, vis_fake2), axis=-1)
                torchvision.utils.save_image(fake_sample2_vis, os.path.join(exp_path, 'sample2_discrete_epoch_{}.png'.format(epoch)), normalize=True)
                
                pred2 = gen_non_diffusive_1to2(real_data1)
                
                x1_t = torch.cat((torch.randn_like(real_data1), pred2), axis=1)
                fake_sample1_tilda = gen_diffusive_1(x1_t , t1, latent_z1)   
                
                vis_pred2 = pred2[:, :3, :, :]
                vis_cycle2 = gen_non_diffusive_2to1(pred2)[:, :3, :, :]
                vis_fake1_tilda = fake_sample1_tilda[:, :3, :, :]
                
                pred2_vis = torch.cat((vis_real1, vis_pred2, vis_cycle2, vis_fake1_tilda), axis=-1)
                torchvision.utils.save_image(pred2_vis, os.path.join(exp_path, 'sample2_translated_epoch_{}.png'.format(epoch)), normalize=True)
            
            # 4. 保存模型内容 (Checkpoint)
            if args.save_content:
                if epoch % args.save_content_every == 0:
                    print('Saving content.')
                    # 保存完整状态字典，方便断点续训
                    content = {'epoch': epoch + 1, 'global_step': global_step, 'args': args,
                            'gen_diffusive_1_dict': gen_diffusive_1.state_dict(), 'optimizer_gen_diffusive_1': optimizer_gen_diffusive_1.state_dict(),
                            'gen_diffusive_2_dict': gen_diffusive_2.state_dict(), 'optimizer_gen_diffusive_2': optimizer_gen_diffusive_2.state_dict(),
                            'scheduler_gen_diffusive_1': scheduler_gen_diffusive_1.state_dict(), 'disc_diffusive_1_dict': disc_diffusive_1.state_dict(),
                            'scheduler_gen_diffusive_2': scheduler_gen_diffusive_2.state_dict(), 'disc_diffusive_2_dict': disc_diffusive_2.state_dict(),
                            'gen_non_diffusive_1to2_dict': gen_non_diffusive_1to2.state_dict(), 'optimizer_gen_non_diffusive_1to2': optimizer_gen_non_diffusive_1to2.state_dict(),
                            'gen_non_diffusive_2to1_dict': gen_non_diffusive_2to1.state_dict(), 'optimizer_gen_non_diffusive_2to1': optimizer_gen_non_diffusive_2to1.state_dict(),
                            'scheduler_gen_non_diffusive_1to2': scheduler_gen_non_diffusive_1to2.state_dict(), 'scheduler_gen_non_diffusive_2to1': scheduler_gen_non_diffusive_2to1.state_dict(),
                            'optimizer_disc_diffusive_1': optimizer_disc_diffusive_1.state_dict(), 'scheduler_disc_diffusive_1': scheduler_disc_diffusive_1.state_dict(),
                            'optimizer_disc_diffusive_2': optimizer_disc_diffusive_2.state_dict(), 'scheduler_disc_diffusive_2': scheduler_disc_diffusive_2.state_dict(),
                            'optimizer_disc_non_diffusive_cycle1': optimizer_disc_non_diffusive_cycle1.state_dict(), 'scheduler_disc_non_diffusive_cycle1': scheduler_disc_non_diffusive_cycle1.state_dict(),
                            'optimizer_disc_non_diffusive_cycle2': optimizer_disc_non_diffusive_cycle2.state_dict(), 'scheduler_disc_non_diffusive_cycle2': scheduler_disc_non_diffusive_cycle2.state_dict(),
                            'disc_non_diffusive_cycle1_dict': disc_non_diffusive_cycle1.state_dict(),'disc_non_diffusive_cycle2_dict': disc_non_diffusive_cycle2.state_dict()}
                    
                    torch.save(content, os.path.join(exp_path, 'content.pth'))
                
            if epoch % args.save_ckpt_every == 0:
                if args.use_ema:
                    # 交换 EMA 参数进行保存
                    optimizer_gen_diffusive_1.swap_parameters_with_ema(store_params_in_ema=True)
                    optimizer_gen_diffusive_2.swap_parameters_with_ema(store_params_in_ema=True)
                    optimizer_gen_non_diffusive_1to2.swap_parameters_with_ema(store_params_in_ema=True)
                    optimizer_gen_non_diffusive_2to1.swap_parameters_with_ema(store_params_in_ema=True)                    
                torch.save(gen_diffusive_1.state_dict(), os.path.join(exp_path, 'gen_diffusive_1_{}.pth'.format(epoch)))
                torch.save(gen_diffusive_2.state_dict(), os.path.join(exp_path, 'gen_diffusive_2_{}.pth'.format(epoch)))
                torch.save(gen_non_diffusive_1to2.state_dict(), os.path.join(exp_path, 'gen_non_diffusive_1to2_{}.pth'.format(epoch)))
                torch.save(gen_non_diffusive_2to1.state_dict(), os.path.join(exp_path, 'gen_non_diffusive_2to1_{}.pth'.format(epoch)))                
                if args.use_ema:
                    # 换回原始参数
                    optimizer_gen_diffusive_1.swap_parameters_with_ema(store_params_in_ema=True)
                    optimizer_gen_diffusive_2.swap_parameters_with_ema(store_params_in_ema=True)
                    optimizer_gen_non_diffusive_1to2.swap_parameters_with_ema(store_params_in_ema=True)
                    optimizer_gen_non_diffusive_2to1.swap_parameters_with_ema(store_params_in_ema=True)

        # 5. 验证集循环 (Validation Loop)
        # 5. 验证集循环 (Validation Loop)
        # -----------------------------------------------------------------
        # 方向 1: Domain 1 -> Domain 2
        for iteration, (x_val , y_val) in enumerate(data_loader_val): 
            real_data = x_val.to(device, non_blocking=True)
            source_data = y_val.to(device, non_blocking=True)
            
            # 【修复】强制切片验证集
            if real_data.shape[1] > 3: real_data = real_data[:, :3, :, :]
            if source_data.shape[1] > 3: source_data = source_data[:, :3, :, :]
            
            x1_t = torch.cat((torch.randn_like(real_data), source_data), axis=1)
            fake_sample1 = sample_from_model(pos_coeff, gen_diffusive_1, args.num_timesteps, x1_t, T, args)            
            
            fake_rgb = fake_sample1[:, :3, :, :]
            real_rgb = real_data[:, :3, :, :]
            
            fake_rgb = to_range_0_1(fake_rgb) ; fake_rgb = fake_rgb/fake_rgb.mean()
            real_rgb = to_range_0_1(real_rgb) ; real_rgb = real_rgb/real_rgb.mean()

            fake_rgb = fake_rgb.cpu().numpy()
            real_rgb = real_rgb.cpu().numpy()
            
            val_l1_loss[0, epoch, iteration] = abs(fake_rgb - real_rgb).mean()
            val_psnr_values[0, epoch, iteration] = psnr(real_rgb, fake_rgb, data_range=real_rgb.max())

        # 方向 2: Domain 2 -> Domain 1
        for iteration, (y_val , x_val) in enumerate(data_loader_val): 
            real_data = x_val.to(device, non_blocking=True)
            source_data = y_val.to(device, non_blocking=True)
            
            # 【修复】强制切片验证集
            if real_data.shape[1] > 3: real_data = real_data[:, :3, :, :]
            if source_data.shape[1] > 3: source_data = source_data[:, :3, :, :]
            
            x2_t = torch.cat((torch.randn_like(real_data), source_data), axis=1)
            fake_sample2 = sample_from_model(pos_coeff, gen_diffusive_2, args.num_timesteps, x2_t, T, args)

            fake_rgb = fake_sample2[:, :3, :, :]
            real_rgb = real_data[:, :3, :, :]
            
            fake_rgb = to_range_0_1(fake_rgb) ; fake_rgb = fake_rgb/fake_rgb.mean()
            real_rgb = to_range_0_1(real_rgb) ; real_rgb = real_rgb/real_rgb.mean()
            
            fake_rgb = fake_rgb.cpu().numpy()
            real_rgb = real_rgb.cpu().numpy()
            
            val_l1_loss[1, epoch, iteration] = abs(fake_rgb - real_rgb).mean()
            val_psnr_values[1, epoch, iteration] = psnr(real_rgb, fake_rgb, data_range=real_rgb.max())

        print(f"Epoch {epoch} PSNR (1->2): {np.nanmean(val_psnr_values[0, epoch, :])}")
        print(f"Epoch {epoch} PSNR (2->1): {np.nanmean(val_psnr_values[1, epoch, :])}")
        np.save('{}/val_l1_loss.npy'.format(exp_path), val_l1_loss)
        np.save('{}/val_psnr_values.npy'.format(exp_path), val_psnr_values)              



# =========================================================================
# 分布式训练初始化 (全局作用域)
# =========================================================================
def cleanup():
    dist.destroy_process_group()  

def init_processes(rank, size, fn, args):
    """ 初始化分布式训练环境 """
    os.environ['MASTER_ADDR'] = args.master_address
    os.environ['MASTER_PORT'] = args.port_num
    torch.cuda.set_device(args.local_rank)
    gpu = args.local_rank
    dist.init_process_group(backend='nccl', init_method='env://', rank=rank, world_size=size)
    fn(rank, gpu, args)
    dist.barrier()
    cleanup()  

# =========================================================================
# 主程序
# =========================================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser('syndiff parameters')
    # ... (保持原本的 ArgumentParser 设置不变) ...
    # 只需要把下面的部分复制过来即可
    
    parser.add_argument('--seed', type=int, default=1024, help='seed used for initialization')
    parser.add_argument('--resume', action='store_true',default=False)
    parser.add_argument('--image_size', type=int, default=256, help='size of image')
    parser.add_argument('--num_channels', type=int, default=3, help='channel of image')
    parser.add_argument('--centered', action='store_false', default=True, help='-1,1 scale')
    parser.add_argument('--use_geometric', action='store_true',default=False)
    parser.add_argument('--beta_min', type=float, default= 0.1, help='beta_min for diffusion')
    parser.add_argument('--beta_max', type=float, default=20., help='beta_max for diffusion')
    parser.add_argument('--num_channels_dae', type=int, default=64, help='number of initial channels in denosing model')
    parser.add_argument('--n_mlp', type=int, default=3, help='number of mlp layers for z')
    parser.add_argument('--ch_mult', nargs='+', type=int, help='channel multiplier')
    parser.add_argument('--num_res_blocks', type=int, default=2, help='number of resnet blocks per scale')
    # ✅ 全局 self-attn 的分辨率（建议只放低分辨率，如 16/8）
    parser.add_argument('--attn_resolutions', nargs='+', type=int, default=[16],
                        help='resolution list for global attention, e.g. --attn_resolutions 16 8')
    # ✅ 局部注意力（用于高分辨率 256/128/64），做消融用
    parser.add_argument('--local_attn_type', type=str, default='none', choices=['none', 'cbam', 'scsa', 'coord'],
                    help='local attention type for ablation')
    parser.add_argument('--local_attn_resolutions', nargs='+', type=int, default=[128, 64, 32],
                    help='resolution list for local attention')


    parser.add_argument('--dropout', type=float, default=0., help='drop-out rate')
    parser.add_argument('--resamp_with_conv', action='store_false', default=True, help='always up/down sampling with conv')
    parser.add_argument('--conditional', action='store_false', default=True, help='noise conditional')
    parser.add_argument('--fir', action='store_false', default=True, help='FIR')
    parser.add_argument('--fir_kernel', default=[1, 3, 3, 1], help='FIR kernel')
    parser.add_argument('--skip_rescale', action='store_false', default=True, help='skip rescale')
    parser.add_argument('--resblock_type', default='biggan', help='tyle of resnet block, choice in biggan and ddpm')
    parser.add_argument('--progressive', type=str, default='none', choices=['none', 'output_skip', 'residual'], help='progressive type for output') 
    parser.add_argument('--progressive_input', type=str, default='residual', choices=['none', 'input_skip', 'residual'], help='progressive type for input')
    parser.add_argument('--progressive_combine', type=str, default='sum', choices=['sum', 'cat'], help='progressive combine method.')
    parser.add_argument('--embedding_type', type=str, default='positional', choices=['positional', 'fourier'], help='type of time embedding')
    parser.add_argument('--fourier_scale', type=float, default=16., help='scale of fourier transform')
    parser.add_argument('--not_use_tanh', action='store_true',default=False)
    parser.add_argument('--exp', default='ixi_synth', help='name of experiment')
    parser.add_argument('--input_path', help='path to input data')
    parser.add_argument('--output_path', help='path to output saves')
    parser.add_argument('--nz', type=int, default=100)
    parser.add_argument('--num_timesteps', type=int, default=4)
    parser.add_argument('--z_emb_dim', type=int, default=256)
    parser.add_argument('--t_emb_dim', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=1, help='input batch size')
    parser.add_argument('--num_epoch', type=int, default=1200)
    parser.add_argument('--ngf', type=int, default=64)
    parser.add_argument('--lr_g', type=float, default=1.5e-4, help='learning rate g')
    parser.add_argument('--lr_d', type=float, default=1e-4, help='learning rate d')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam')
    parser.add_argument('--beta2', type=float, default=0.9, help='beta2 for adam')
    parser.add_argument('--no_lr_decay',action='store_true', default=False)
    parser.add_argument('--use_ema', action='store_true', default=False, help='use EMA or not')
    parser.add_argument('--ema_decay', type=float, default=0.9999, help='decay rate for EMA')
    parser.add_argument('--r1_gamma', type=float, default=0.05, help='coef for r1 reg')
    parser.add_argument('--lazy_reg', type=int, default=None, help='lazy regulariation.')
    parser.add_argument('--save_content', action='store_true',default=False)
    parser.add_argument('--save_content_every', type=int, default=40, help='save content for resuming every x epochs')
    parser.add_argument('--save_ckpt_every', type=int, default=20, help='save ckpt every x epochs')
    parser.add_argument('--lambda_l1_loss', type=float, default=2.0, help='weightening of l1 loss part of diffusion ans cycle models')
    parser.add_argument('--num_proc_node', type=int, default=1, help='The number of nodes in multi node env.')
    parser.add_argument('--num_process_per_node', type=int, default=1, help='number of gpus')
    parser.add_argument('--node_rank', type=int, default=0, help='The index of node.')
    parser.add_argument('--local_rank', type=int, default=0, help='rank of process in the node')
    parser.add_argument('--master_address', type=str, default='127.0.0.1', help='address for master')
    parser.add_argument('--contrast1', type=str, default='T1', help='contrast selection for model')
    parser.add_argument('--contrast2', type=str, default='T2', help='contrast selection for model')
    parser.add_argument('--port_num', type=str, default='6021', help='port selection for code')
   
    args = parser.parse_args()
    
    args.world_size = args.num_proc_node * args.num_process_per_node
    size = args.num_process_per_node

    if size > 1:
        processes = []
        for rank in range(size):
            args.local_rank = rank
            global_rank = rank + args.node_rank * args.num_process_per_node
            global_size = args.num_proc_node * args.num_process_per_node
            args.global_rank = global_rank
            
            print('Node rank %d, local proc %d, global proc %d' % (args.node_rank, rank, global_rank))
            
            p = Process(target=init_processes, args=(global_rank, global_size, train_syndiff, args))
            p.start()
            processes.append(p)
            
        for p in processes:
            p.join()
    else:
        init_processes(0, size, train_syndiff, args)
