import cv2
import os

# 读取原始 mask
mask_path = "/mnt/mydata/gww/sj/syndiff/SynDiff/新建文件夹/masks/2.png" 
img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

# 像素值乘以 100，这样 1 变成 100(灰色)，2 变成 200(亮灰)，0 还是 0(黑)
vis_img = img * 100 

# 保存到当前目录看一眼
cv2.imwrite("check_vis.png", vis_img)
print("已生成 check_vis.png，请下载查看，你应该能看到灰色的斑块了。")