import torch
# 导入刚才保存的文件
from my_model import UNetplus 

# 1. 初始化模型
# 【注意】看你原来的代码，inDim可能是4 (RGB+Vessel)，也可能是3 (RGB)
# 如果你只输入图片，就设为 3。如果报错通道对不上，就改成 4。
model = UNetplus(inDim=4) 

# 2. 你的权重路径
weights_path = "/mnt/mydata/gww/sj/syndiff/SynDiff/best_model_final.pth"
full_state_dict = torch.load(weights_path, map_location='cpu')

# 3. 【核心步骤】清洗权重 Key
# 你的 .pth 是大模型存的，Key 可能是 "UNetplus.ConLevel00..."
# 或者是 "module.UNetplus.ConLevel00..."
# 我们要把前面的 "UNetplus." 去掉，只留后面的给我们的模型用
new_state_dict = {}
for k, v in full_state_dict.items():
    if "UNetplus." in k:
        # 找到 "UNetplus." 后面那部分
        # 例如 "preEMnet.UNetplus.ConLevel00.weight" -> "ConLevel00.weight"
        name = k.split("UNetplus.")[1]
        new_state_dict[name] = v

# 4. 加载清洗后的权重
try:
    model.load_state_dict(new_state_dict)
    print("✅ 终于成功了！UNetplus 权重加载完毕！")
except Exception as e:
    print(f"❌ 还是有点问题: {e}")
    # 如果报错 mismatch，看看是不是 inDim 设错了
    # 比如 input channel 是 3 还是 4

# 5. 简单测试
model.eval()
# 如果你 inDim=4，这里就要用 4
dummy_input = torch.randn(1, 4, 512, 512) 
with torch.no_grad():
    out = model(dummy_input)
print(f"✅ 输出形状: {out.shape}") # 应该是 [1, 5, 512, 512] (5类病灶)