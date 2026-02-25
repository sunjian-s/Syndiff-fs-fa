import torch
import torch.nn as nn
import torch.nn.functional as F

class DynamicSnakeConv(nn.Module):
    def __init__(self, inc, outc, kernel_size=5):
        super().__init__()
        self.kernel_size = kernel_size
        self.inc = inc
        self.outc = outc
        
        # 1. 偏移量预测
        self.offset_conv = nn.Conv2d(inc, 3 * kernel_size, kernel_size=1, bias=True)
        
        # 2. 最终聚合卷积 (保持结构不变，以便兼容权重)
        # 形状: [outc, inc*k, 1, 1]
        self.final_conv = nn.Conv2d(inc * kernel_size, outc, kernel_size=1, bias=True)
        
        self._init_weights()

    def _init_weights(self):
        nn.init.constant_(self.offset_conv.weight, 0)
        nn.init.constant_(self.offset_conv.bias, 0)

    def forward(self, x):
        B, C, H, W = x.shape
        device = x.device
        
        # 1. 预测偏移
        offsets = self.offset_conv(x)
        
        dx = torch.tanh(offsets[:, :self.kernel_size, :, :])
        dy = torch.tanh(offsets[:, self.kernel_size:2*self.kernel_size, :, :])
        mask = torch.sigmoid(offsets[:, 2*self.kernel_size:, :, :])

        # 2. 基础网格
        y_grid, x_grid = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')
        x_grid = x_grid.unsqueeze(0).repeat(B, 1, 1).float()
        y_grid = y_grid.unsqueeze(0).repeat(B, 1, 1).float()
        
        x_grid_norm = 2.0 * x_grid / (W - 1) - 1.0
        y_grid_norm = 2.0 * y_grid / (H - 1) - 1.0

        # ==========================================
        # 3. 【核心优化】省显存计算模式
        # ==========================================
        # 不再创建 stored_features 列表，而是直接累加结果
        output_accumulator = 0
        
        # 将 final_conv 的权重切片，分段计算
        # weight shape: [Out, In*K, 1, 1] -> 切成 K 份 [Out, In, 1, 1]
        weight_chunks = torch.chunk(self.final_conv.weight, self.kernel_size, dim=1)
        
        for i in range(self.kernel_size):
            # A. 坐标计算
            xi = x_grid_norm + dx[:, i, :, :] * (2.0 / (W - 1)) * 5.0
            yi = y_grid_norm + dy[:, i, :, :] * (2.0 / (H - 1)) * 5.0
            grid = torch.stack([xi, yi], dim=-1)
            
            # B. 采样
            sampled_feat = F.grid_sample(x, grid, mode='bilinear', padding_mode='reflection', align_corners=True)
            
            # C. 掩码加权
            sampled_feat = sampled_feat * mask[:, i:i+1, :, :]
            
            # D. 【即时卷积】直接与对应的权重切片做卷积，不保存中间大张量
            # 使用 F.conv2d 手动计算
            # sampled_feat: [B, C, H, W]
            # weight_chunks[i]: [Out, C, 1, 1]
            slice_out = F.conv2d(sampled_feat, weight_chunks[i], bias=None, stride=1, padding=0)
            
            # E. 累加
            output_accumulator = output_accumulator + slice_out

        # 4. 最后加上 bias
        if self.final_conv.bias is not None:
            output_accumulator = output_accumulator + self.final_conv.bias.view(1, -1, 1, 1)

        return output_accumulator