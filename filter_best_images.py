import os
import shutil
import torch
from PIL import Image
from torchvision import transforms
from my_model import UNetplus  # 直接导入模型定义

# ================= 配置区域 =================
# 1. 你生成图片的文件夹 (训练完后，图片会在 results/exp_... 里面)
#    注意：这里要填具体的 Epoch 文件夹，或者把所有生成的图拷到一个文件夹里
GEN_DIR = "./results/exp_fundus_ffa_mask" 

# 2. 筛选出的好图存到哪里
OUT_DIR = "./results/best_images_filtered"

# 3. 权重路径
WEIGHTS = "/mnt/mydata/gww/sj/syndiff/SynDiff/best_model_final.pth"

# 4. 阈值 (总响应值超过多少才算好图？先设低点，跑一次看看再调)
THRESHOLD_SCORE = 10.0 
# ===========================================

def make_4_channel(x):
    # x shape: [1, 3, H, W]
    green = x[:, 1:2, :, :]
    return torch.cat([x, green], dim=1)

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.makedirs(OUT_DIR, exist_ok=True)

    # 1. 加载裁判模型
    print("正在加载裁判模型...")
    model = UNetplus(inDim=4).to(device)
    
    # 加载权重 (带清洗 Key 的逻辑)
    state_dict = torch.load(WEIGHTS, map_location='cpu')
    new_state_dict = {}
    for k, v in state_dict.items():
        if "UNetplus." in k:
            new_state_dict[k.split("UNetplus.")[1]] = v
        else:
            new_state_dict[k] = v
    
    # 有时候权重不完全匹配 strict=False 更稳妥
    model.load_state_dict(new_state_dict, strict=False)
    model.eval()

    # 2. 预处理
    # 训练生成的 png 图片通常已经是 [0, 255]，ToTensor 会自动转成 [0, 1]
    # 这正是裁判想要的范围，所以不需要像训练代码那样 (x+1)/2
    to_tensor = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor() 
    ])

    print(f"开始筛选文件夹: {GEN_DIR}")
    count = 0

    # 3. 遍历图片
    for root, dirs, files in os.walk(GEN_DIR):
        for img_name in files:
            if not img_name.endswith(('.png', '.jpg')):
                continue
            
            # 过滤掉真实的图 (通常带有 real 字样)，我们只评测生成的图 (fake/sample)
            if 'real' in img_name: 
                continue

            img_path = os.path.join(root, img_name)
            
            try:
                img = Image.open(img_path).convert('RGB')
                img_t = to_tensor(img).unsqueeze(0).to(device) # [1, 3, 512, 512]
                
                # 构造4通道输入
                img_input = make_4_channel(img_t)
                
                with torch.no_grad():
                    # 裁判输出 [1, 5, 512, 512]
                    # 注意：这里我们用 Softmax 输出，因为用来打分，概率图更直观
                    prob_map = model(img_input)
                
                # 4. 计算分数
                # 取出 1-4 通道 (病灶)，求和
                lesion_map = prob_map[:, 1:, :, :]
                score = lesion_map.sum().item()
                
                # 打印一些信息帮你确定阈值
                # print(f"{img_name}: Score = {score:.2f}")

                # 5. 筛选
                if score > THRESHOLD_SCORE:
                    print(f"✅ 发现好图: {img_name} (分值: {score:.2f})")
                    shutil.copy(img_path, os.path.join(OUT_DIR, f"score_{int(score)}_{img_name}"))
                    count += 1
                    
            except Exception as e:
                print(f"跳过坏图 {img_name}: {e}")

    print(f"筛选结束！共找到 {count} 张好图，保存在 {OUT_DIR}")

if __name__ == "__main__":
    main()