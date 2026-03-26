class VGG19PerceptualLoss(nn.Module):
    def __init__(
        self,
        device,
        use_layer4=False,
        layer_weights=None,
    ):
        super().__init__()
        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features
        self.slice1 = nn.Sequential(*list(vgg.children())[:4])    # relu1_2
        self.slice2 = nn.Sequential(*list(vgg.children())[4:9])   # relu2_2
        self.slice3 = nn.Sequential(*list(vgg.children())[9:18])  # relu3_4
        self.use_layer4 = use_layer4
        if self.use_layer4:
            self.slice4 = nn.Sequential(*list(vgg.children())[18:27])  # relu4_4

        for param in self.parameters():
            param.requires_grad = False

        if layer_weights is None:
            if self.use_layer4:
                self.layer_weights = [1.0, 1.0, 0.5, 0.25]
            else:
                self.layer_weights = [1.0, 1.0, 0.5]
        else:
            self.layer_weights = layer_weights

        self.register_buffer(
            "mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

        self.to(device)

    def _preprocess(self, x):
        # 假设输入范围 [-1, 1]
        x = (x + 1.0) / 2.0
        x = torch.clamp(x, 0.0, 1.0)

        # 单通道复制成 3 通道
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)
        elif x.size(1) > 3:
            x = x[:, :3, :, :]

        x = (x - self.mean) / self.std
        return x

    def forward(self, gen_img, real_img):
        x = self._preprocess(gen_img)
        y = self._preprocess(real_img)

        h_x1 = self.slice1(x)
        h_y1 = self.slice1(y)

        h_x2 = self.slice2(h_x1)
        h_y2 = self.slice2(h_y1)

        h_x3 = self.slice3(h_x2)
        h_y3 = self.slice3(h_y2)

        loss = (
            self.layer_weights[0] * F.l1_loss(h_x1, h_y1) +
            self.layer_weights[1] * F.l1_loss(h_x2, h_y2) +
            self.layer_weights[2] * F.l1_loss(h_x3, h_y3)
        )

        if self.use_layer4:
            h_x4 = self.slice4(h_x3)
            h_y4 = self.slice4(h_y3)
            loss = loss + self.layer_weights[3] * F.l1_loss(h_x4, h_y4)

        return loss
# ============================================
# 【新增】1. 小波损失函数 (Wavelet Loss)
import torch
import torch.nn as nn
import torch.nn.functional as F

class WaveletLoss(nn.Module):
    def __init__(self, device='cuda'):
        super(WaveletLoss, self).__init__()
        self.device = device

    def forward(self, input, target):
    # 兼容1/3通道转灰度
        def to_gray(x):
            if x.shape[1] == 1:
                return x  # 1通道直接返回
            elif x.shape[1] == 3:
                return 0.299 * x[:,0,:,:] + 0.587 * x[:,1,:,:] + 0.114 * x[:,2,:,:]
            else:
                raise ValueError(f"不支持的通道数：{x.shape[1]}")
        
        input_gray = to_gray(input)
        target_gray = to_gray(target)
        input_gray = input_gray.unsqueeze(1)
        target_gray = target_gray.unsqueeze(1)

        # 2. 提取低频 (LL)
        input_LL = F.avg_pool2d(input_gray, kernel_size=2, stride=2)
        target_LL = F.avg_pool2d(target_gray, kernel_size=2, stride=2)
        
        # 3. 【核心修正】必须使用 bilinear，严禁使用 nearest 产生棋盘格伪影
        input_up = F.interpolate(input_LL, scale_factor=2, mode='bilinear', align_corners=False)
        target_up = F.interpolate(target_LL, scale_factor=2, mode='bilinear', align_corners=False)
        
        # 4. 提取高频 (High)
        input_high = input_gray - input_up
        target_high = target_gray - target_up

        # 5. 计算绝对误差
        loss_low = F.l1_loss(input_LL, target_LL)
        loss_high = F.l1_loss(input_high, target_high)

        # 6. 权重融合
        weight_low = 0.5
        weight_high = 10.0 
        
        return weight_low * loss_low + weight_high * loss_high

# ============================================
# 【新增】2. 亮度加权损失 (LFSG的核心 - 亮度部分)
# ============================================
import torch
import torch.nn as nn

class LuminanceWeightedLoss(nn.Module):
    """高亮/极暗极值惩罚：附带 FOV 背景拦截防御"""
    def __init__(self, alpha=10.0, is_target_ffa=True, bg_threshold=-0.9):
        super().__init__()
        self.alpha = alpha
        self.is_target_ffa = is_target_ffa
        # 设定背景判定阈值：在 [-1, 1] 空间中，小于 -0.9 的统统视为无意义黑边
        self.bg_threshold = bg_threshold 

    def forward(self, pred, target):
        # 1. 映射掩码空间：[-1, 1] -> [0, 1]
        target_norm = (target + 1.0) * 0.5 
        
        # 2. 提取 FOV 掩码（逻辑非黑边区域为 1，黑边区域为 0）
        # 这里必须用 detach() 防止梯度回传干扰阈值判定
        fov_mask = (target > self.bg_threshold).float().detach()
        
        # 3. 计算基础权重矩阵
        if self.is_target_ffa:
            # FFA 域：白亮病灶加权。背景本来就是黑的 (0^4=0)，不会被错误放大，但加上 FOV 拦截更严谨
            weight_mask = 1.0 + self.alpha * (target_norm ** 4.0).detach() 
        else:
            # Fundus 域：暗色病灶加权。
            weight_mask = 1.0 + self.alpha * ((1.0 - target_norm) ** 4.0).detach()
            
            # 【核心纠偏逻辑】：利用 FOV 掩码，强行把黑边背景的异常高权重拍回 1.0
            # 只有在 fov_mask > 0（即真实的眼底圆形区域内），才应用暴击权重；黑边区域只保留基础 L1 权重 1.0
            weight_mask = torch.where(fov_mask > 0, weight_mask, torch.ones_like(weight_mask))
            
        # 4. 计算加权误差
        diff = torch.abs(pred - target)
        return torch.mean(diff * weight_mask)