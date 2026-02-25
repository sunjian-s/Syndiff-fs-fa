import torch
import torch.nn as nn
from my_model import UNetplus 

class LesionJudgeLoss(nn.Module):
    def __init__(self, weights_path, device='cuda'):
        super().__init__()
        print("🔍 初始化裁判 Loss 模块 (Feature Matching 版)...")
        
        # 1. 实例化模型
        self.judge = UNetplus(inDim=4)
        
        # 2. 加载权重
        print(f"📥 加载权重: {weights_path}")
        full_state_dict = torch.load(weights_path, map_location='cpu')
        new_state_dict = {}
        for k, v in full_state_dict.items():
            if "UNetplus." in k:
                name = k.split("UNetplus.")[1]
                new_state_dict[name] = v
        self.judge.load_state_dict(new_state_dict)
        self.judge.to(device)
        self.judge.eval()
        
        # 3. 冻结参数
        for param in self.judge.parameters():
            param.requires_grad = False
            
        self.device = device
        self.l2_loss = nn.MSELoss()
        
        # ==========================================================
        # 【黑科技】使用 Hook 提取中间层特征
        # 我们不看最后的 Softmax (全是0)，我们要看中间的 ConLevel40 (特征丰富)
        # ==========================================================
        self.features = {}
        
        def get_activation(name):
            def hook(model, input, output):
                self.features[name] = output
            return hook

        # 注册钩子：钩住 ConLevel40 这一层 (这是 U-Net 最底部的深层语义特征)
        # 如果报错 AttributeError，请检查 my_model.py 里有没有 self.ConLevel40
        self.judge.ConLevel40.register_forward_hook(get_activation('bottleneck'))

    def make_4_channel(self, x):
        # x: [B, 3, H, W] -> [0, 1]
        green_channel = x[:, 1:2, :, :] 
        return torch.cat([x, green_channel], dim=1)

    def forward(self, generated_img, real_img):
        # 1. 归一化 [-1, 1] -> [0, 1]
        gen_01 = (generated_img + 1.0) / 2.0
        real_01 = (real_img + 1.0) / 2.0

        # 2. 构造输入
        gen_input = self.make_4_channel(gen_01)
        real_input = self.make_4_channel(real_01)
        
        # 3. 裁判前向传播 (Hook 会自动把特征存进 self.features)
        
        # (A) 跑真实图
        with torch.no_grad():
            _ = self.judge(real_input) # 不需要返回值，我们要的是 hook 里的东西
            real_feat = self.features['bottleneck'].clone() # 取出特征 [B, 512, H/16, W/16]
            
        # (B) 跑生成图
        _ = self.judge(gen_input)
        gen_feat = self.features['bottleneck'] # 取出特征
        
        # 4. 计算 Feature Loss
        # 特征图通常数值在 0~10 之间，非常健康
        loss = self.l2_loss(gen_feat, real_feat)

        # 5. 适当放大权重 (因为特征层维度高，MSE均值可能会小)
        loss = loss * 50000.0 

        # [DEBUG] 看看现在的 Loss 是不是正常了
        if loss.item() < 1e-4: # 如果还很小，打印出来看看
            print(f"[DEBUG] Feature Loss: {loss.item():.6f}")
        
        return loss