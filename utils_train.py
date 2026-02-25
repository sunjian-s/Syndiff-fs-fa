import logging
import os
import matplotlib.pyplot as plt
from datetime import datetime

# ============================
# 1. 日志工具 (Logger)
# ============================
def get_logger(save_dir):
    """
    创建一个 logger，同时输出到控制台和文件
    """
    logger = logging.getLogger("TrainLog")
    logger.setLevel(logging.INFO)
    
    # 防止重复添加 handler
    if not logger.handlers:
        # 文件输出
        file_handler = logging.FileHandler(os.path.join(save_dir, 'train_log.txt'))
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
        logger.addHandler(file_handler)
        
        # 控制台输出
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(console_handler)
        
    return logger

def make_exp_dir(base_dir="./experiments"):
    """
    根据当前时间创建实验文件夹，如 ./experiments/20260125_1830_SnakeConv
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join(base_dir, f"run_{timestamp}")
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "checkpoints"), exist_ok=True) # 权重保存目录
    os.makedirs(os.path.join(exp_dir, "samples"), exist_ok=True)     # 采样图保存目录
    return exp_dir

# ============================
# 2. 绘图工具 (Loss Plotter)
# ============================
class LossTracker:
    def __init__(self, save_dir):
        self.save_dir = save_dir
        self.epochs = []
        self.train_losses = []
        self.val_losses = [] # 如果你有验证集的话
        
    def update(self, epoch, train_loss, val_loss=None):
        self.epochs.append(epoch)
        self.train_losses.append(train_loss)
        if val_loss is not None:
            self.val_losses.append(val_loss)
        
        self.plot()

    def plot(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.epochs, self.train_losses, label='Train Loss', color='blue', linewidth=2)
        if self.val_losses:
            plt.plot(self.epochs, self.val_losses, label='Val Loss', color='orange', linestyle='--', linewidth=2)
            
        plt.title('Training Loss Curve')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 保存图片，覆盖旧的，实现“动态更新”
        plt.savefig(os.path.join(self.save_dir, 'loss_curve.png'))
        plt.close()