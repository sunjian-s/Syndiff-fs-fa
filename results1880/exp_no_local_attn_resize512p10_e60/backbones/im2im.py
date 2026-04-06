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

# 【注意】: 这份代码依赖于您上一个文件 (discriminator.py) 中的 
# get_norm_layer, init_net, 和 NLayerDiscriminator, ResnetBlock
# 但这里为了完整性，重新定义了 ResnetGenerator 和 ResnetBlock

def define_G(input_nc=1, output_nc=1, ngf=64, netG='resnet_9blocks', norm='batch', use_dropout=False, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """
    【工厂函数】: 创建并初始化一个生成器 (Generator)
    
    这是您在 `train.py` 中会调用的高级函数 (例如 `gen_non_diffusive_1to2`)。
    
    Parameters:
        input_nc (int) -- 输入通道数 (例如 Fundus=3)
        output_nc (int) -- 输出通道数 (例如 FFA=3)
        ngf (int) -- 基础通道数 (控制模型“宽度”)
        netG (str) -- 架构名称: 'resnet_9blocks', 'resnet_6blocks', 'unet_256', 'unet_128'
        norm (str) -- 归一化层类型: 'batch', 'instance', 'none'
        ...
        gpu_ids (int list) -- 在哪些 GPU 上运行
    Returns:
        一个初始化好的生成器网络
    """
    net = None
    # 从上一个文件 (discriminator.py) 中获取 get_norm_layer 函数
    # (假设) from .discriminator import get_norm_layer
    norm_layer = get_norm_layer(norm_type=norm) 

    if netG == 'resnet_9blocks':
        # 如果选择 resnet_9blocks，则实例化 ResnetGenerator 并传入9个块
        net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9)
    elif netG == 'resnet_6blocks':
        # 如果选择 resnet_6blocks，则实例化 ResnetGenerator 并传入6个块
        net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=6)
    elif netG == 'unet_128':
        # (UnetGenerator 的定义不在此文件中)
        net = UnetGenerator(input_nc, output_nc, 7, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    elif netG == 'unet_256':
        # (UnetGenerator 的定义不在此文件中)
        net = UnetGenerator(input_nc, output_nc, 8, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % netG)
    
    # (假设) from .discriminator import init_net
    # 使用 init_net 函数来初始化网络 (放到GPU, 初始化权重)
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
            input_nc (int)      -- 输入通道数
            output_nc (int)     -- 输出通道数
            ngf (int)           -- 基础通道数
            norm_layer          -- 归一化层
            use_dropout (bool)  -- 是否在 ResNet 块中使用 Dropout
            n_blocks (int)      -- ResNet 块的数量
            padding_type (str)  -- 填充类型: 'reflect', 'replicate', 'zero'
        """
        assert(n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        # 检查是否使用偏置 (Bias)。如果使用 InstanceNorm，通常不使用偏置。
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        # --- 1. 编码器 (Encoder) ---
        
        # 初始卷积层 (7x7 卷积)
        model = [nn.ReflectionPad2d(3), # 反射填充，减少边缘伪影
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2 # 2 次下采样
        for i in range(n_downsampling):  # 添加 2 个下采样层
            mult = 2 ** i # 第一次: mult=1, 第二次: mult=2
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]
            # 第一次: (ngf, ngf*2)
            # 第二次: (ngf*2, ngf*4)

        # --- 2. 转换器 (Transformer) ---
        mult = 2 ** n_downsampling # mult = 4
        for i in range(n_blocks):       # 添加 N 个 ResNet 块
            # 在这里，特征图大小不变 (例如 64x64)，只改变通道数
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        # --- 3. 解码器 (Decoder) ---
        for i in range(n_downsampling):  # 添加 2 个上采样层
            mult = 2 ** (n_downsampling - i) # 第一次: mult=4, 第二次: mult=2
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
            # 第一次: (ngf*4, ngf*2)
            # 第二次: (ngf*2, ngf)

        # 最终输出层 (7x7 卷积)
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()] # 将输出压缩到 [-1, 1] 范围

        self.model = nn.Sequential(*model) # 将所有层打包成一个模型

    def forward(self, input):
        """前向传播"""
        return self.model(input)


class ResnetBlock(nn.Module):
    """定义一个 Resnet 块 (在生成器中间使用)"""

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """初始化 Resnet 块"""
        super(ResnetBlock, self).__init__()
        # 真正的主体是一个“卷积块”
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """
        构造一个包含两个卷积层的“卷积块”。
        (Conv -> Norm -> ReLU -> [Dropout] -> Conv -> Norm)
        """
        conv_block = []
        p = 0
        
        # --- 第一个卷积层 ---
        if padding_type == 'reflect': # 反射填充
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate': # 复制填充
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
        # 注意：第二个卷积层后面没有 ReLU
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """前向传播 (带跳跃连接)"""
        # 【关键】将输入 x 与 块的输出 相加。这就是“残差” (Res) 的含义。
        out = x + self.conv_block(x)  
        return out