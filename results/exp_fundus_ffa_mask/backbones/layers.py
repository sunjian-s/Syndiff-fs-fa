# ---------------------------------------------------------------
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
# ... (版权和许可信息) ...
# ---------------------------------------------------------------

# pylint: skip-file
''' Codes adapted from https://github.com/yang-song/score_sde_pytorch/blob/main/models/ncsnpp.py
'''
import math
import string
from functools import partial
import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np


def get_act(config):
  """【辅助】从配置文件 (config) 中获取激活函数。"""

  if config.model.nonlinearity.lower() == 'elu':
    return nn.ELU()
  elif config.model.nonlinearity.lower() == 'relu':
    return nn.ReLU()
  elif config.model.nonlinearity.lower() == 'lrelu':
    return nn.LeakyReLU(negative_slope=0.2)
  elif config.model.nonlinearity.lower() == 'swish':
    return nn.SiLU() # Swish/SiLU 是现代扩散模型中常用的激活函数
  else:
    raise NotImplementedError('activation function does not exist!')


def ncsn_conv1x1(in_planes, out_planes, stride=1, bias=True, dilation=1, init_scale=1., padding=0):
  """【工厂】1x1 卷积层 (NCSN 风格)。使用简单的缩放初始化。"""
  conv = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=bias, dilation=dilation,
                   padding=padding)
  init_scale = 1e-10 if init_scale == 0 else init_scale
  conv.weight.data *= init_scale
  conv.bias.data *= init_scale
  return conv


def variance_scaling(scale, mode, distribution,
                     in_axis=1, out_axis=0,
                     dtype=torch.float32,
                     device='cpu'):
  """
  【核心初始化】从 JAX 移植过来的方差缩放初始化器。
  这是 'default_init' 的核心实现。
  """

  def _compute_fans(shape, in_axis=1, out_axis=0):
    """计算 fan_in 和 fan_out"""
    receptive_field_size = np.prod(shape) / shape[in_axis] / shape[out_axis]
    fan_in = shape[in_axis] * receptive_field_size
    fan_out = shape[out_axis] * receptive_field_size
    return fan_in, fan_out

  def init(shape, dtype=dtype, device=device):
    fan_in, fan_out = _compute_fans(shape, in_axis, out_axis)
    if mode == "fan_in":
      denominator = fan_in
    elif mode == "fan_out":
      denominator = fan_out
    elif mode == "fan_avg":
      denominator = (fan_in + fan_out) / 2
    else:
      raise ValueError(
        "invalid mode for variance scaling initializer: {}".format(mode))
    variance = scale / denominator # 计算方差
    if distribution == "normal":
      # 正态分布初始化
      return torch.randn(*shape, dtype=dtype, device=device) * np.sqrt(variance)
    elif distribution == "uniform":
      # 均匀分布初始化
      return (torch.rand(*shape, dtype=dtype, device=device) * 2. - 1.) * np.sqrt(3 * variance)
    else:
      raise ValueError("invalid distribution for variance scaling initializer")

  return init # 返回那个内部的 init 函数


def default_init(scale=1.):
  """【初始化器】DDPM 中使用的标准初始化方法。"""
  scale = 1e-10 if scale == 0 else scale
  # 使用 fan_avg (输入/输出平均) 和 均匀分布
  return variance_scaling(scale, 'fan_avg', 'uniform')


class Dense(nn.Module):
  """【已弃用】一个本应使用 default_init 的线性层，但未实现。"""
  def __init__(self):
    super().__init__()


def ddpm_conv1x1(in_planes, out_planes, stride=1, bias=True, init_scale=1., padding=0):
  """【工厂】1x1 卷积层 (DDPM 风格)，使用 'default_init' 初始化。"""
  conv = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=padding, bias=bias)
  conv.weight.data = default_init(init_scale)(conv.weight.data.shape)
  nn.init.zeros_(conv.bias)
  return conv


def ncsn_conv3x3(in_planes, out_planes, stride=1, bias=True, dilation=1, init_scale=1., padding=1):
  """【工厂】3x3 卷积层 (NCSN 风格)，使用简单的缩放初始化。"""
  init_scale = 1e-10 if init_scale == 0 else init_scale
  conv = nn.Conv2d(in_planes, out_planes, stride=stride, bias=bias,
                   dilation=dilation, padding=padding, kernel_size=3)
  conv.weight.data *= init_scale
  conv.bias.data *= init_scale
  return conv


def ddpm_conv3x3(in_planes, out_planes, stride=1, bias=True, dilation=1, init_scale=1., padding=1):
  """【工厂】3x3 卷积层 (DDPM 风格)，使用 'default_init' 初始化。"""
  conv = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=padding,
                   dilation=dilation, bias=bias)
  conv.weight.data = default_init(init_scale)(conv.weight.data.shape)
  nn.init.zeros_(conv.bias)
  return conv

  ###########################################################################
  # 以下函数移植自 NCSNv1/NCSNv2 代码库
  ###########################################################################


class CRPBlock(nn.Module):
  """
  【模块】Chained Residual Pooling (CRP) 块 - 链式残差池化
  
  这是一种用于语义分割的模块，它通过多次池化和卷积来聚合“多尺度上下文”。
  """
  def __init__(self, features, n_stages, act=nn.ReLU(), maxpool=True):
    super().__init__()
    self.convs = nn.ModuleList()
    for i in range(n_stages):
      self.convs.append(ncsn_conv3x3(features, features, stride=1, bias=False))
    self.n_stages = n_stages
    if maxpool:
      self.pool = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)
    else:
      self.pool = nn.AvgPool2d(kernel_size=5, stride=1, padding=2)

    self.act = act

  def forward(self, x):
    x = self.act(x)
    path = x
    for i in range(self.n_stages):
      path = self.pool(path) # 1. 池化
      path = self.convs[i](path) # 2. 卷积
      x = path + x # 3. 将不同尺度的结果加回原始 x
    return x


class CondCRPBlock(nn.Module):
  """【模块】带条件 (Conditional) 的 CRP 块"""
  def __init__(self, features, n_stages, num_classes, normalizer, act=nn.ReLU()):
    super().__init__()
    self.convs = nn.ModuleList()
    self.norms = nn.ModuleList()
    self.normalizer = normalizer # 归一化层 (例如 AdaGN)
    for i in range(n_stages):
      # 【关键】在卷积之前有一个“条件归一化”层
      self.norms.append(normalizer(features, num_classes, bias=True))
      self.convs.append(ncsn_conv3x3(features, features, stride=1, bias=False))

    self.n_stages = n_stages
    self.pool = nn.AvgPool2d(kernel_size=5, stride=1, padding=2)
    self.act = act

  def forward(self, x, y): # y 是条件 (例如时间 t 或类别)
    x = self.act(x)
    path = x
    for i in range(self.n_stages):
      path = self.norms[i](path, y) # 【关键】在这里注入条件 y
      path = self.pool(path)
      path = self.convs[i](path)
      x = path + x
    return x


class RCUBlock(nn.Module):
  """
  【模块】Residual Conv Unit (RCU) - 残差卷积单元
  
  这是一个包含多个残差块的“大”残差块。
  """
  def __init__(self, features, n_blocks, n_stages, act=nn.ReLU()):
    super().__init__()

    # setattr 是一种动态添加层的方法
    for i in range(n_blocks):
      for j in range(n_stages):
        setattr(self, '{}_{}_conv'.format(i + 1, j + 1), ncsn_conv3x3(features, features, stride=1, bias=False))

    self.stride = 1
    self.n_blocks = n_blocks
    self.n_stages = n_stages
    self.act = act

  def forward(self, x):
    for i in range(self.n_blocks):
      residual = x # 保存外部跳跃连接
      for j in range(self.n_stages):
        x = self.act(x)
        x = getattr(self, '{}_{}_conv'.format(i + 1, j + 1))(x)

      x += residual # 添加外部残差
    return x


class CondRCUBlock(nn.Module):
  """【模块】带条件 (Conditional) 的 RCU 块"""
  def __init__(self, features, n_blocks, n_stages, num_classes, normalizer, act=nn.ReLU()):
    super().__init__()

    for i in range(n_blocks):
      for j in range(n_stages):
        # 【关键】在卷积之前有“条件归一化”
        setattr(self, '{}_{}_norm'.format(i + 1, j + 1), normalizer(features, num_classes, bias=True))
        setattr(self, '{}_{}_conv'.format(i + 1, j + 1), ncsn_conv3x3(features, features, stride=1, bias=False))

    self.stride = 1
    self.n_blocks = n_blocks
    self.n_stages = n_stages
    self.act = act
    self.normalizer = normalizer

  def forward(self, x, y): # y 是条件
    for i in range(self.n_blocks):
      residual = x
      for j in range(self.n_stages):
        x = getattr(self, '{}_{}_norm'.format(i + 1, j + 1))(x, y) # 注入条件 y
        x = self.act(x)
        x = getattr(self, '{}_{}_conv'.format(i + 1, j + 1))(x)

      x += residual
    return x


class MSFBlock(nn.Module):
  """
  【模块】Multi-Scale Fusion (MSF) 块 - 多尺度融合
  
  用于 U-Net 的“跳跃连接”，它接收来自编码器多个尺度的特征图 (xs)，
  将它们融合（上采样并相加）成一个。
  """
  def __init__(self, in_planes, features):
    super().__init__()
    assert isinstance(in_planes, list) or isinstance(in_planes, tuple)
    self.convs = nn.ModuleList()
    self.features = features # 融合后的目标通道数

    for i in range(len(in_planes)):
      # 为每个尺度的输入创建一个 3x3 卷积
      self.convs.append(ncsn_conv3x3(in_planes[i], features, stride=1, bias=True))

  def forward(self, xs, shape): # xs 是一个特征图列表, shape 是目标 H,W
    sums = torch.zeros(xs[0].shape[0], self.features, *shape, device=xs[0].device)
    for i in range(len(self.convs)):
      h = self.convs[i](xs[i])
      # 【关键】将所有尺度的特征图上采样到相同的目标 'shape'
      h = F.interpolate(h, size=shape, mode='bilinear', align_corners=True)
      sums += h # 将它们相加
    return sums


class CondMSFBlock(nn.Module):
  """【模块】带条件 (Conditional) 的 MSF 块"""
  def __init__(self, in_planes, features, num_classes, normalizer):
    super().__init__()
    assert isinstance(in_planes, list) or isinstance(in_planes, tuple)

    self.convs = nn.ModuleList()
    self.norms = nn.ModuleList()
    self.features = features
    self.normalizer = normalizer

    for i in range(len(in_planes)):
      # 【关键】在卷积之前有“条件归一化”
      self.convs.append(ncsn_conv3x3(in_planes[i], features, stride=1, bias=True))
      self.norms.append(normalizer(in_planes[i], num_classes, bias=True))

  def forward(self, xs, y, shape): # y 是条件
    sums = torch.zeros(xs[0].shape[0], self.features, *shape, device=xs[0].device)
    for i in range(len(self.convs)):
      h = self.norms[i](xs[i], y) # 注入条件 y
      h = self.convs[i](h)
      h = F.interpolate(h, size=shape, mode='bilinear', align_corners=True)
      sums += h
    return sums


class RefineBlock(nn.Module):
  """
  【模块】RefineBlock (精炼块)
  
  NCSN/RefineNet 中的一个核心 U-Net 块，它结合了 RCU, MSF 和 CRP。
  """
  def __init__(self, in_planes, features, act=nn.ReLU(), start=False, end=False, maxpool=True):
    super().__init__()

    assert isinstance(in_planes, tuple) or isinstance(in_planes, list)
    self.n_blocks = n_blocks = len(in_planes)

    # 1. 适配层 (Adaptation Convolutions)
    self.adapt_convs = nn.ModuleList()
    for i in range(n_blocks):
      self.adapt_convs.append(RCUBlock(in_planes[i], 2, 2, act))

    # 3. 输出卷积
    self.output_convs = RCUBlock(features, 3 if end else 1, 2, act)

    if not start:
      # 2. 多尺度融合 (Multi-Scale Fusion)
      self.msf = MSFBlock(in_planes, features)

    # (此块的 CRP 似乎被移除了，或者在原版NCSNpp中有)
    self.crp = CRPBlock(features, 2, act, maxpool=maxpool)

  def forward(self, xs, output_shape):
    assert isinstance(xs, tuple) or isinstance(xs, list)
    hs = []
    # 1. 先用 RCU 适配所有输入
    for i in range(len(xs)):
      h = self.adapt_convs[i](xs[i])
      hs.append(h)

    # 2. 融合 (如果有多个输入)
    if self.n_blocks > 1:
      h = self.msf(hs, output_shape)
    else:
      h = hs[0]

    h = self.crp(h) # 3. 链式残差池化
    h = self.output_convs(h) # 4. 输出 RCU

    return h


class CondRefineBlock(nn.Module):
  """【模块】带条件 (Conditional) 的 RefineBlock"""
  def __init__(self, in_planes, features, num_classes, normalizer, act=nn.ReLU(), start=False, end=False):
    super().__init__()
    # ... (结构与 RefineBlock 相同) ...
    assert isinstance(in_planes, tuple) or isinstance(in_planes, list)
    self.n_blocks = n_blocks = len(in_planes)

    self.adapt_convs = nn.ModuleList()
    for i in range(n_blocks):
      # 【关键】使用带条件的 RCU
      self.adapt_convs.append(
        CondRCUBlock(in_planes[i], 2, 2, num_classes, normalizer, act)
      )
    # 【关键】使用带条件的 RCU
    self.output_convs = CondRCUBlock(features, 3 if end else 1, 2, num_classes, normalizer, act)

    if not start:
      # 【关键】使用带条件的 MSF
      self.msf = CondMSFBlock(in_planes, features, num_classes, normalizer)

    # 【关键】使用带条件的 CRP
    self.crp = CondCRPBlock(features, 2, num_classes, normalizer, act)

  def forward(self, xs, y, output_shape): # y 是条件
    assert isinstance(xs, tuple) or isinstance(xs, list)
    hs = []
    # 1. 适配 (注入条件 y)
    for i in range(len(xs)):
      h = self.adapt_convs[i](xs[i], y)
      hs.append(h)

    # 2. 融合 (注入条件 y)
    if self.n_blocks > 1:
      h = self.msf(hs, y, output_shape)
    else:
      h = hs[0]

    h = self.crp(h, y) # 3. CRP (注入条件 y)
    h = self.output_convs(h, y) # 4. 输出 (注入条件 y)

    return h


class ConvMeanPool(nn.Module):
  """【下采样】先卷积，再平均池化 (Conv -> AvgPool)"""
  def __init__(self, input_dim, output_dim, kernel_size=3, biases=True, adjust_padding=False):
    super().__init__()
    if not adjust_padding:
      conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride=1, padding=kernel_size // 2, bias=biases)
      self.conv = conv
    else:
      conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride=1, padding=kernel_size // 2, bias=biases)
      self.conv = nn.Sequential(
        nn.ZeroPad2d((1, 0, 1, 0)),
        conv
      )

  def forward(self, inputs):
    output = self.conv(inputs)
    # 【关键】通过像素错位求和来实现 2x2 平均池化 (一种老式但高效的写法)
    output = sum([output[:, :, ::2, ::2], output[:, :, 1::2, ::2],
                  output[:, :, ::2, 1::2], output[:, :, 1::2, 1::2]]) / 4.
    return output


class MeanPoolConv(nn.Module):
  """【下采样】先平均池化，再卷积 (AvgPool -> Conv)"""
  def __init__(self, input_dim, output_dim, kernel_size=3, biases=True):
    super().__init__()
    self.conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride=1, padding=kernel_size // 2, bias=biases)

  def forward(self, inputs):
    output = inputs
    # 1. 平均池化
    output = sum([output[:, :, ::2, ::2], output[:, :, 1::2, ::2],
                  output[:, :, ::2, 1::2], output[:, :, 1::2, 1::2]]) / 4.
    return self.conv(output) # 2. 卷积


class UpsampleConv(nn.Module):
  """【上采样】使用 PixelShuffle (像素重排) 实现上采样"""
  def __init__(self, input_dim, output_dim, kernel_size=3, biases=True):
    super().__init__()
    self.conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride=1, padding=kernel_size // 2, bias=biases)
    self.pixelshuffle = nn.PixelShuffle(upscale_factor=2)

  def forward(self, inputs):
    output = inputs
    # 1. 将通道数 x4 (B,C,H,W) -> (B,C*4,H,W)
    output = torch.cat([output, output, output, output], dim=1)
    # 2. PixelShuffle: (B,C*4,H,W) -> (B,C,H*2,W*2)
    output = self.pixelshuffle(output)
    return self.conv(output) # 3. 卷积


class ResidualBlock(nn.Module):
  """
  【模块】(NCSN 风格的) 残差块
  
  这是一个更老的、用于 NCSN 的残差块定义。
  """
  def __init__(self, input_dim, output_dim, resample=None, act=nn.ELU(),
               normalization=nn.InstanceNorm2d, adjust_padding=False, dilation=1):
    super().__init__()
    self.non_linearity = act
    self.input_dim = input_dim
    self.output_dim = output_dim
    self.resample = resample # 'down' 或 None
    self.normalization = normalization
    
    # --- 根据 resample (是否下采样) 来定义不同的层 ---
    if resample == 'down':
      if dilation > 1:
        self.conv1 = ncsn_conv3x3(input_dim, input_dim, dilation=dilation)
        self.normalize2 = normalization(input_dim)
        self.conv2 = ncsn_conv3x3(input_dim, output_dim, dilation=dilation)
        conv_shortcut = partial(ncsn_conv3x3, dilation=dilation)
      else:
        # 标准下采样块
        self.conv1 = ncsn_conv3x3(input_dim, input_dim)
        self.normalize2 = normalization(input_dim)
        self.conv2 = ConvMeanPool(input_dim, output_dim, 3, adjust_padding=adjust_padding)
        conv_shortcut = partial(ConvMeanPool, kernel_size=1, adjust_padding=adjust_padding)

    elif resample is None: # 不改变分辨率
      if dilation > 1:
        conv_shortcut = partial(ncsn_conv3x3, dilation=dilation)
        self.conv1 = ncsn_conv3x3(input_dim, output_dim, dilation=dilation)
        self.normalize2 = normalization(output_dim)
        self.conv2 = ncsn_conv3x3(output_dim, output_dim, dilation=dilation)
      else:
        # 标准残差块
        conv_shortcut = partial(ncsn_conv1x1)
        self.conv1 = ncsn_conv3x3(input_dim, output_dim)
        self.normalize2 = normalization(output_dim)
        self.conv2 = ncsn_conv3x3(output_dim, output_dim)
    else:
      raise Exception('invalid resample value')

    # 【跳跃连接】如果维度变化 (通道数或分辨率)，需要一个 1x1 卷积来匹配
    if output_dim != input_dim or resample is not None:
      self.shortcut = conv_shortcut(input_dim, output_dim)

    self.normalize1 = normalization(input_dim)

  def forward(self, x):
    # 经典的“预激活”(pre-activation)残差块
    output = self.normalize1(x)
    output = self.non_linearity(output)
    output = self.conv1(output)
    output = self.normalize2(output)
    output = self.non_linearity(output)
    output = self.conv2(output)

    if self.output_dim == self.input_dim and self.resample is None:
      shortcut = x # 如果维度相同，跳跃连接就是 x 本身
    else:
      shortcut = self.shortcut(x) # 否则，跳跃连接需要通过 1x1 卷积

    return shortcut + output


###########################################################################
# 以下函数移植自 DDPM 代码库:
#  https://github.com/hojonathanho/diffusion/blob/master/diffusion_tf/nn.py
###########################################################################

def get_timestep_embedding(timesteps, embedding_dim, max_positions=10000):
  """
  【核心】时间步嵌入 (Sinusoidal Positional Embedding)
  
  这是 "Attention Is All You Need" 论文中提出的标准位置编码。
  它将一个整数时间步 t (例如 500) 转换为一个高维向量 (例如 128 维)。
  """
  assert len(timesteps.shape) == 1  # 确保输入是 [B]
  half_dim = embedding_dim // 2
  # 1. 计算 embedding: emb = log(10000) / (D/2 - 1)
  emb = math.log(max_positions) / (half_dim - 1)
  # 2. 计算 emb = exp(-[0, 1, ..., D/2-1] * emb)
  emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
  # 3. emb = t * emb (广播)
  emb = timesteps.float()[:, None] * emb[None, :]
  # 4. [sin(emb), cos(emb)]
  emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
  if embedding_dim % 2 == 1:  # 如果是奇数，补一个 0
    emb = F.pad(emb, (0, 1), mode='constant')
  assert emb.shape == (timesteps.shape[0], embedding_dim)
  return emb


def _einsum(a, b, c, x, y):
  """【辅助】torch.einsum (爱因斯坦求和约定) 的包装器"""
  einsum_str = '{},{}->{}'.format(''.join(a), ''.join(b), ''.join(c))
  return torch.einsum(einsum_str, x, y)


def contract_inner(x, y):
  """【辅助】实现 tensordot(x, y, 1)，即矩阵乘法"""
  x_chars = list(string.ascii_lowercase[:len(x.shape)])
  y_chars = list(string.ascii_lowercase[len(x.shape):len(y.shape) + len(x.shape)])
  y_chars[0] = x_chars[-1]  # 确保 y 的第一个轴与 x 的最后一个轴相乘
  out_chars = x_chars[:-1] + y_chars[1:]
  return _einsum(x_chars, y_chars, out_chars, x, y)


class NIN(nn.Module):
  """
  【模块】Network In Network (NIN)
  
  本质上是一个 1x1 卷积层，但它是用全连接层 (Linear) 和维度重排 (permute) 来实现的。
  """
  def __init__(self, in_dim, num_units, init_scale=0.1):
    super().__init__()
    self.W = nn.Parameter(default_init(scale=init_scale)((in_dim, num_units)), requires_grad=True)
    self.b = nn.Parameter(torch.zeros(num_units), requires_grad=True)

  def forward(self, x):
    # x: (B, C, H, W)
    x = x.permute(0, 2, 3, 1) # -> (B, H, W, C)
    y = contract_inner(x, self.W) + self.b # 矩阵乘法: (B,H,W,C) x (C, C_out) -> (B,H,W,C_out)
    return y.permute(0, 3, 1, 2) # -> (B, C_out, H, W)


class AttnBlock(nn.Module):
  """【模块】(DDPM 风格的) 自注意力块 (Self-Attention Block)"""
  def __init__(self, channels):
    super().__init__()
    self.GroupNorm_0 = nn.GroupNorm(num_groups=32, num_channels=channels, eps=1e-6)
    self.NIN_0 = NIN(channels, channels) # Q (Query)
    self.NIN_1 = NIN(channels, channels) # K (Key)
    self.NIN_2 = NIN(channels, channels) # V (Value)
    self.NIN_3 = NIN(channels, channels, init_scale=0.) # 最终输出

  def forward(self, x):
    B, C, H, W = x.shape
    h = self.GroupNorm_0(x)
    q = self.NIN_0(h)
    k = self.NIN_1(h)
    v = self.NIN_2(h)

    # 计算 Q * K^T (bchw, bcij -> bhwij)
    w = torch.einsum('bchw,bcij->bhwij', q, k) * (int(C) ** (-0.5)) # 缩放
    w = torch.reshape(w, (B, H, W, H * W))
    w = F.softmax(w, dim=-1) # (H, W, H*W)
    w = torch.reshape(w, (B, H, W, H, W))
    
    # 计算 (Q * K^T) * V (bhwij, bcij -> bchw)
    h = torch.einsum('bhwij,bcij->bchw', w, v)
    h = self.NIN_3(h)
    return x + h # 残差连接


class Upsample(nn.Module):
  """【上采样】(DDPM 风格) 先插值，再卷积"""
  def __init__(self, channels, with_conv=False):
    super().__init__()
    if with_conv:
      self.Conv_0 = ddpm_conv3x3(channels, channels)
    self.with_conv = with_conv

  def forward(self, x):
    B, C, H, W = x.shape
    # 1. 最近邻插值 (H, W) -> (H*2, W*2)
    h = F.interpolate(x, (H * 2, W * 2), mode='nearest')
    if self.with_conv:
      h = self.Conv_0(h) # 2. (可选) 3x3 卷积平滑
    return h


class Downsample(nn.Module):
  """【下采样】(DDPM 风格)"""
  def __init__(self, channels, with_conv=False):
    super().__init__()
    if with_conv:
      # 带步长(stride=2)的卷积来实现下采样
      self.Conv_0 = ddpm_conv3x3(channels, channels, stride=2, padding=0)
    self.with_conv = with_conv

  def forward(self, x):
    B, C, H, W = x.shape
    if self.with_conv:
      x = F.pad(x, (0, 1, 0, 1)) # 模拟 'SAME' 填充
      x = self.Conv_0(x)
    else:
      # 使用平均池化下采样
      x = F.avg_pool2d(x, kernel_size=2, stride=2, padding=0)

    assert x.shape == (B, C, H // 2, W // 2)
    return x


class ResnetBlockDDPM(nn.Module):
  """
  【模块】(DDPM 风格的) 残差块
  
  这是现代扩散模型 (如 DDPM, diffusers 库) 中最标准的残差块。
  它包含了时间嵌入 (temb) 的注入。
  """
  def __init__(self, act, in_ch, out_ch=None, temb_dim=None, conv_shortcut=False, dropout=0.1):
    super().__init__()
    if out_ch is None:
      out_ch = in_ch
    self.GroupNorm_0 = nn.GroupNorm(num_groups=32, num_channels=in_ch, eps=1e-6)
    self.act = act
    self.Conv_0 = ddpm_conv3x3(in_ch, out_ch) # 第一个卷积
    if temb_dim is not None:
      # 【关键】时间嵌入 t 的全连接层
      self.Dense_0 = nn.Linear(temb_dim, out_ch)
      self.Dense_0.weight.data = default_init()(self.Dense_0.weight.data.shape)
      nn.init.zeros_(self.Dense_0.bias)

    self.GroupNorm_1 = nn.GroupNorm(num_groups=32, num_channels=out_ch, eps=1e-6)
    self.Dropout_0 = nn.Dropout(dropout)
    self.Conv_1 = ddpm_conv3x3(out_ch, out_ch, init_scale=0.) # 第二个卷积 (初始化为 0)
    
    # 【跳跃连接】
    if in_ch != out_ch: # 如果通道数不同
      if conv_shortcut: # 使用 3x3 卷积
        self.Conv_2 = ddpm_conv3x3(in_ch, out_ch)
      else: # 使用 1x1 卷积 (NIN)
        self.NIN_0 = NIN(in_ch, out_ch)
    self.out_ch = out_ch
    self.in_ch = in_ch
    self.conv_shortcut = conv_shortcut

  def forward(self, x, temb=None):
    B, C, H, W = x.shape
    assert C == self.in_ch
    out_ch = self.out_ch if self.out_ch else self.in_ch
    h = self.act(self.GroupNorm_0(x)) # 预激活
    h = self.Conv_0(h)
    
    # 【关键】注入时间步 t
    if temb is not None:
      # (Dense(act(temb)) -> [B, out_ch])
      # [:, :, None, None] -> [B, out_ch, 1, 1]
      # 通过广播加到特征图上
      h += self.Dense_0(self.act(temb))[:, :, None, None]
      
    h = self.act(self.GroupNorm_1(h))
    h = self.Dropout_0(h)
    h = self.Conv_1(h)
    
    # 计算跳跃连接
    if C != out_ch:
      if self.conv_shortcut:
        x = self.Conv_2(x)
      else:
        x = self.NIN_0(x)
        
    return x + h # 残差相加