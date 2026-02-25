# ---------------------------------------------------------------
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
# ... (版权和许可信息，大部分代码源自 pix2pix 和 CycleGAN) ...
# ---------------------------------------------------------------

# coding=utf-8
# ... (Google Research 版权信息) ...

# pylint: skip-file
''' Codes adapted from https://github.com/yang-song/score_sde_pytorch/blob/main/models/ncsnpp.py
'''

from . import utils, layers, layerspp, dense_layer # 导入其他自定义模块
import torch.nn as nn
import functools
import torch
import numpy as np
from torch.nn import init


class Identity(nn.Module):
    """一个占位符层，它什么也不做，只是原样返回输入。"""
    def forward(self, x):
        return x


def get_norm_layer(norm_type='instance'):
    """
    【工厂函数】: 获取一个归一化层 (Normalization Layer)。
    
    Parameters:
        norm_type (str) -- 'batch' (BatchNorm), 'instance' (InstanceNorm), 或 'none'.
    
    - 'batch' (批量归一化): 适用于大多数任务，但受批量大小影响。
    - 'instance' (实例归一化): 适用于风格迁移和 GANs，因为它独立处理每张图像。
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        def norm_layer(x): return Identity() # 如果为 'none'，则返回占位符
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def init_weights(net, init_type='normal', init_gain=0.02):
    """
    【核心】初始化网络权重。
    
    一个好的权重初始化对 GAN 的稳定训练至关重要。
    
    Parameters:
        net (network)   -- 要初始化的网络
        init_type (str) -- 初始化方法: 'normal', 'xavier', 'kaiming', 'orthogonal'
        init_gain (float)  -- 缩放因子
    """
    def init_func(m):  # 定义一个内部函数，用于递归地应用到网络的每一层 (m)
        classname = m.__class__.__name__
        # 检查这一层是否是 Conv 或 Linear，并且有权重
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            # 根据选择的类型，使用不同的 PyTorch 内置初始化器
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            # 同时将偏置 (bias) 初始化为 0
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        # 单独处理 BatchNorm 层
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # 递归地将 init_func 应用到网络 (net) 的所有子模块


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """
    【辅助函数】: 初始化一个网络。
    
    这个函数做了两件事：
    1. 将网络放到正确的 GPU 上（并可选地用 DataParallel 封装以支持多卡）。
    2. 调用 init_weights 来初始化权重。
    """
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0]) # 将网络移动到第一张指定的 GPU
        net = torch.nn.DataParallel(net, gpu_ids)  # 使用 DataParallel 封装以实现多 GPU
    init_weights(net, init_type, init_gain=init_gain)
    return net

def define_D(input_nc=1, ndf=64, which_model_netD='basic',n_layers_D=3, norm='instance', use_sigmoid=False, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """
    【工厂函数】: 定义和初始化一个“判别器” (Discriminator)。
    
    这是您在 `train.py` 中会调用的高级函数。
    """
    netD = None
    norm_layer = get_norm_layer(norm_type=norm)

    if which_model_netD == 'basic': # 'basic' 是 PatchGAN 的默认类型
        netD = NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer, use_sigmoid=use_sigmoid, gpu_ids=gpu_ids)
    elif which_model_netD == 'n_layers': # 允许指定层数
        netD = NLayerDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer, use_sigmoid=use_sigmoid, gpu_ids=gpu_ids)
    elif which_model_netD == 'pixel': # 一种逐像素判断的判别器
        netD = PixelDiscriminator(input_nc, ndf, norm_layer=norm_layer, use_sigmoid=use_sigmoid, gpu_ids=gpu_ids)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' %
                                  which_model_netD)
    # 返回初始化好的网络 (已放到 GPU 并初始化权重)
    return init_net(netD, init_type, init_gain, gpu_ids)



def define_G(input_nc=1, output_nc=1, ngf=64, netG='resnet_9blocks', norm='instance', use_dropout=False, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """
    【工厂函数】: 定义和初始化一个“生成器” (Generator)。
    
    这是您在 `train.py` 中调用的高级函数 (例如 `gen_non_diffusive_1to2`)。
    
    Parameters:
        ...
        netG (str) -- 架构名称: 'resnet_9blocks' (标准 CycleGAN) 或 'resnet_6blocks' (轻量版)
        ...
    """
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netG == 'resnet_9blocks':
        net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9)
    elif netG == 'resnet_6blocks':
        net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=6)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % netG)
    # 返回初始化好的网络
    return init_net(net, init_type, init_gain, gpu_ids)



class ResnetGenerator(nn.Module):
    """
    【核心】Resnet (残差网络) 生成器架构。
    
    这是 CycleGAN 的标准生成器。
    它是一个“编码器-转换器-解码器”结构。
    """

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=9, padding_type='reflect'):
        """
        构造一个 Resnet 生成器
        
        Parameters:
            ...
            n_blocks (int) -- 中间 ResNet 块的数量 (9块是标准，6块是轻量版)
            ...
        """
        assert(n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        # 检查是否使用偏置 (Bias)。如果使用 InstanceNorm，通常不使用偏置。
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        # --- 1. 编码器 (Encoder) ---
        # 初始卷积层
        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):  # 添加 2 个下采样层
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        # --- 2. 转换器 (Transformer) ---
        mult = 2 ** n_downsampling
        for i in range(n_blocks):       # 添加 N 个 ResNet 块
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        # --- 3. 解码器 (Decoder) ---
        for i in range(n_downsampling):  # 添加 2 个上采样层 (使用转置卷积)
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        
        # 最终输出层
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()] # 将输出压缩到 [-1, 1] 范围

        self.model = nn.Sequential(*model)

    def forward(self, input):
        """Standard forward"""
        return self.model(input)


class ResnetBlock(nn.Module):
    """定义一个 Resnet 块 (在生成器中间使用)"""

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """初始化 Resnet 块"""
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """构造一个包含两个卷积层的“卷积块”"""
        conv_block = []
        p = 0
        
        # --- 第一个卷积层 ---
        if padding_type == 'reflect': # 反射填充 (更适合图像，避免边缘伪影)
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate': # 复制边缘填充
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero': # 零填充
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        # --- 第二个卷积层 ---
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """前向传播 (带跳跃连接)"""
        # 【关键】将输入 x 与 块的输出 相加。这就是“残差” (Res) 的含义。
        out = x + self.conv_block(x)  
        return out
    


# 定义 PatchGAN 判别器
class NLayerDiscriminator(nn.Module):
    """
    【核心】N 层 PatchGAN 判别器。
    
    它不是对整张图像输出一个“真/假”分数，而是对图像中的 N x N 个“图块”(Patch)
    分别输出“真/假”分数，这能保留更多高频细节。
    """
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False, gpu_ids=[]):
        super(NLayerDiscriminator, self).__init__()
        self.gpu_ids = gpu_ids
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4 # 卷积核大小
        padw = 1 # 填充
        # 判别器的第一层 (输入 -> ndf)
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]

        # 中间层 (逐渐增加通道数)
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers): # n_layers 默认为 3
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8) # 通道数乘数 (1 -> 2 -> 4)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        # 倒数第二层
        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8) # (4 -> 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        # 最终输出层 (输出一个 1 通道的分数图)
        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]

        if use_sigmoid: # 如果使用标准的 GAN 损失 (BCE)，则需要 Sigmoid
            sequence += [nn.Sigmoid()] # (在 SynDiff 中使用 MSE Loss，所以不需要)

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """前向传播"""
        return self.model(input)