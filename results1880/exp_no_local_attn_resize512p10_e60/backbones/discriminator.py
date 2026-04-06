# ---------------------------------------------------------------
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
# ... (版权和许可信息) ...
# ---------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# 导入自定义模块
from . import up_or_down_sampling # 导入上/下采样函数
from . import dense_layer # 导入我们之前分析过的 dense 和 conv2d 工厂函数
from . import layers # 导入时间步嵌入函数

# 从自定义模块中分配函数，方便调用
dense = dense_layer.dense
conv2d = dense_layer.conv2d
get_sinusoidal_positional_embedding = layers.get_timestep_embedding

class TimestepEmbedding(nn.Module):
    """
    时间步嵌入模块 (Time Embedding)
    
    将一个标量时间步 't' (例如 500) 转换为一个高维向量 (例如 128 维)。
    这是为了让神经网络能够“理解”时间。
    """
    def __init__(self, embedding_dim, hidden_dim, output_dim, act=nn.LeakyReLU(0.2)):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim

        # 一个简单的 MLP (多层感知机)
        self.main = nn.Sequential(
            dense(embedding_dim, hidden_dim), # 线性层 1
            act, # 激活函数
            dense(hidden_dim, output_dim), # 线性层 2
        )

    def forward(self, temp):
        # 1. 将标量 't' 转换为“正弦位置编码” (Sinusoidal Positional Embedding)
        temb = get_sinusoidal_positional_embedding(temp, self.embedding_dim)
        # 2. 将编码通过 MLP 进一步处理
        temb = self.main(temb)
        return temb
#%%
class DownConvBlock(nn.Module):
    """
    判别器 (D) 和生成器 (G) 中使用的“下采样卷积块” (Downsampling Block)。
    这是一个带残差连接 (Residual Connection) 和时间步嵌入的块。
    """
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size=3,
        padding=1,
        t_emb_dim = 128, # 时间嵌入向量的维度
        downsample=False, # 是否执行下采样（缩小图像）
        act = nn.LeakyReLU(0.2),
        fir_kernel=(1, 3, 3, 1) # 用于平滑下采样的 FIR 滤波器核
    ):
        super().__init__()
    
        
        self.fir_kernel = fir_kernel
        self.downsample = downsample
        
        # 第一个卷积层 (输入通道 -> 输出通道)
        self.conv1 = nn.Sequential(
                     conv2d(in_channel, out_channel, kernel_size, padding=padding),
                     )

        
        # 第二个卷积层 (输出通道 -> 输出通道)
        self.conv2 = nn.Sequential(
                     conv2d(out_channel, out_channel, kernel_size, padding=padding,init_scale=0.) # init_scale=0. 是一个技巧，确保残差块在初始化时是“恒等”的
                     )
        # 这是一个线性层，用于将“时间嵌入”t_emb 映射到与图像通道数(out_channel)匹配的维度
        self.dense_t1= dense(t_emb_dim, out_channel)


        self.act = act
        
        
        # “跳跃连接” (Skip Connection)
        # 如果通道数或尺寸变化，它使用 1x1 卷积来匹配维度
        self.skip = nn.Sequential(
                     conv2d(in_channel, out_channel, 1, padding=0, bias=False),
                     )
        
        

    def forward(self, input, t_emb):
        
        out = self.act(input)
        out = self.conv1(out)
        
        # 【关键】将时间信息注入网络
        # 1. self.dense_t1(t_emb) 将时间 t 转换为 [B, out_channel]
        # 2. [..., None, None] 将其重塑为 [B, out_channel, 1, 1]
        # 3. 通过广播 (broadcasting) 将时间信息“加”到图像的每个像素上
        out += self.dense_t1(t_emb)[..., None, None]
        
        out = self.act(out)
        
        # 如果设置了下采样
        if self.downsample:
            # 使用 FIR 滤波器进行高质量下采样 (缩小图像)
            out = up_or_down_sampling.downsample_2d(out, self.fir_kernel, factor=2)
            # 【残差连接】也必须对原始输入 input 进行同样的下采样
            input = up_or_down_sampling.downsample_2d(input, self.fir_kernel, factor=2)
            
        out = self.conv2(out)
        
        
        skip = self.skip(input) # 计算跳跃连接
        out = (out + skip) / np.sqrt(2) # 【关键】残差相加，并缩放以保持方差稳定


        return out
    
class Discriminator_small(nn.Module):
  """用于小图像 (CIFAR10) 的时间相关判别器。"""

  def __init__(self, nc = 3, ngf = 64, t_emb_dim = 128, act=nn.LeakyReLU(0.2)):
    super().__init__()
    # nc = 输入通道数 (例如 CIFAR10 是 3)
    # ngf = 基础通道数 (控制模型大小)
    self.act = act
    
    
    # 实例化时间嵌入 MLP
    self.t_embed = TimestepEmbedding(
        embedding_dim=t_emb_dim,
        hidden_dim=t_emb_dim,
        output_dim=t_emb_dim,
        act=act,
        )
    
    
    
    # 编码层 (图像分辨率不断降低)
    self.start_conv = conv2d(nc,ngf*2,1, padding=0) # 初始卷积
    self.conv1 = DownConvBlock(ngf*2, ngf*2, t_emb_dim = t_emb_dim,act=act)
    
    self.conv2 = DownConvBlock(ngf*2, ngf*4,  t_emb_dim = t_emb_dim, downsample=True,act=act) # 32x32 -> 16x16
    
    
    self.conv3 = DownConvBlock(ngf*4, ngf*8,  t_emb_dim = t_emb_dim, downsample=True,act=act) # 16x16 -> 8x8

    
    self.conv4 = DownConvBlock(ngf*8, ngf*8, t_emb_dim = t_emb_dim, downsample=True,act=act) # 8x8 -> 4x4
    
    
    self.final_conv = conv2d(ngf*8 + 1, ngf*8, 3,padding=1, init_scale=0.)
    self.end_linear = dense(ngf*8, 1) # 最终输出一个分数 (1 维)
    
    # --- Mini-batch Standard Deviation (StyleGAN2 中的技巧) ---
    # 这是一种高级的 GAN 稳定技巧，它让判别器能“看到”批次中其他图像的统计信息
    # 从而防止生成器“模式崩溃”（只生成一种图像）
    self.stddev_group = 4
    self.stddev_feat = 1
    
        
  def forward(self, x, t, x_t):
    # x = 预测的 x_0 (假) 或 真实的 x_0 (真)
    # t = 时间步
    # x_t = 当前的噪点图 (用于 SynDiff)
    
    # 1. 计算时间嵌入
    t_embed = self.act(self.t_embed(t))   
    
    # 2. 【关键】将 (x_0, x_t) 拼接在一起作为输入
    #    判别器被要求判断：(x_0, x_t) 这一对在 t 时刻是否“真实”
    input_x = torch.cat((x, x_t), dim = 1)
    
    # 3. 通过下采样卷积层
    h0 = self.start_conv(input_x)
    h1 = self.conv1(h0,t_embed)      
    h2 = self.conv2(h1,t_embed)   
    h3 = self.conv3(h2,t_embed)
    out = self.conv4(h3,t_embed)
    
    # 4. 【关键】应用 Mini-batch Standard Deviation
    batch, channel, height, width = out.shape
    group = min(batch, self.stddev_group)
    # 计算批次内特征的“标准差”
    stddev = out.view(
            group, -1, self.stddev_feat, channel // self.stddev_feat, height, width
        )
    stddev = torch.sqrt(stddev.var(0, unbiased=False) + 1e-8)
    stddev = stddev.mean([2, 3, 4], keepdims=True).squeeze(2)
    stddev = stddev.repeat(group, 1, height, width)
    # 将“标准差特征图”拼接到主特征图上
    out = torch.cat([out, stddev], 1)
    
    # 5. 最终处理
    out = self.final_conv(out)
    out = self.act(out)
    
    # 6. 全局求和池化 (Sum Pooling)
    out = out.view(out.shape[0], out.shape[1], -1).sum(2)
    # 7. 最终线性层，输出一个分数
    out = self.end_linear(out)
    
    return out


class Discriminator_large(nn.Module):
  """用于大图像 (CelebA, LSUN, 您的 256x256 图像) 的时间相关判别器。"""

  def __init__(self, nc = 1, ngf = 32, t_emb_dim = 128, act=nn.LeakyReLU(0.2)):
    # nc = 1: 假设输入是 1 通道（灰度图），这匹配了您的 .mat (256x152) 数据
    # ngf = 32: 基础通道数
    super().__init__()
    self.act = act
    
    self.t_embed = TimestepEmbedding( # 时间嵌入 MLP
            embedding_dim=t_emb_dim,
            hidden_dim=t_emb_dim,
            output_dim=t_emb_dim,
            act=act,
        )
      
    # 【关键】这个判别器有更多的下采样层 (6层)，以处理 256x256 的大图
    self.start_conv = conv2d(nc,ngf*2,1, padding=0) # 256x256
    self.conv1 = DownConvBlock(ngf*2, ngf*4, t_emb_dim = t_emb_dim, downsample = True, act=act) # 256 -> 128
    
    self.conv2 = DownConvBlock(ngf*4, ngf*8,  t_emb_dim = t_emb_dim, downsample=True,act=act) # 128 -> 64

    self.conv3 = DownConvBlock(ngf*8, ngf*8,  t_emb_dim = t_emb_dim, downsample=True,act=act) # 64 -> 32
    
    self.conv4 = DownConvBlock(ngf*8, ngf*8, t_emb_dim = t_emb_dim, downsample=True,act=act) # 32 -> 16
    self.conv5 = DownConvBlock(ngf*8, ngf*8, t_emb_dim = t_emb_dim, downsample=True,act=act) # 16 -> 8
    self.conv6 = DownConvBlock(ngf*8, ngf*8, t_emb_dim = t_emb_dim, downsample=True,act=act) # 8 -> 4

 
    self.final_conv = conv2d(ngf*8 + 1, ngf*8, 3,padding=1) # +1 是因为 Mini-batch StdDev
    self.end_linear = dense(ngf*8, 1) # 最终输出 1 个分数
    
    self.stddev_group = 4
    self.stddev_feat = 1
    
        
  def forward(self, x, t, x_t):
    # x = 预测的 x_0 (假) 或 真实的 x_0 (真)
    # t = 时间步
    # x_t = 当前的噪点图
    
    t_embed = self.act(self.t_embed(t))   
    
    # 同样，将 (x_0, x_t) 拼接在一起
    # 【注意】您之前的代码是 x1_pos_sample, t1, x1_tp1.detach()
    # 这里的 forward 签名 (x, t, x_t) 可能与您训练循环中的调用不匹配！
    # (根据您之前的训练循环代码，这里的 'x' 实际上是 x_t，'x_t' 实际上是 x_{t+1})
    input_x = torch.cat((x, x_t), dim = 1) 
    
    # 通过所有 6 个下采样层
    h = self.start_conv(input_x)
    h = self.conv1(h,t_embed)    
    h = self.conv2(h,t_embed)
    h = self.conv3(h,t_embed)
    h = self.conv4(h,t_embed)
    h = self.conv5(h,t_embed)
    out = self.conv6(h,t_embed)
    
    # 应用 Mini-batch Standard Deviation (与 small 版完全相同)
    batch, channel, height, width = out.shape
    group = min(batch, self.stddev_group)
    stddev = out.view(
            group, -1, self.stddev_feat, channel // self.stddev_feat, height, width
        )
    stddev = torch.sqrt(stddev.var(0, unbiased=False) + 1e-8)
    stddev = stddev.mean([2, 3, 4], keepdims=True).squeeze(2)
    stddev = stddev.repeat(group, 1, height, width)
    out = torch.cat([out, stddev], 1)
    
    # 最终层
    out = self.final_conv(out)
    out = self.act(out)
    
    out = out.view(out.shape[0], out.shape[1], -1).sum(2) # 全局求和池化
    out = self.end_linear(out) # 最终分数
    
    return out