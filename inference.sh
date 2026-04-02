#!/bin/bash

# ================= 配置区域 =================
# 1. 模型路径 (建议用 content.pth)
# MODEL_PATH="/mnt/mydata/gww/sj/syndiff/SynDiff/results/exp_fundus_ffa_mask1/gen_diffusive_2_300.pth"
  MODEL_PATH="results/gen_diffusive_2_200.pth"


# 2. 原始大图路径
INPUT_IMG="/mnt/mydata/gww/sj/syndiff/SynDiff/test_set/fundus/55.jpg"

# 3. (可选) 真值大图路径，如果没有可以删掉这行或留空
TARGET_IMG="/mnt/mydata/gww/sj/syndiff/SynDiff/test_set/ffa/55.jpg"

# 4. 输出目录
OUTPUT_DIR="./final_results_baseline+lum+walve"
#OUTPUT_DIR="./final_-baseline"
# ===========================================

export CUDA_VISIBLE_DEVICES=3
# python 1111.py \
python inference2.py \
  --model_path "$MODEL_PATH" \
  --input_image "$INPUT_IMG" \
  --target_image "$TARGET_IMG" \
  --output_path "$OUTPUT_DIR" \
  --image_size 256 \
  --num_channels 3 \
  --num_timesteps 4 \
  --num_res_blocks 2 \
  --ch_mult 1 1 2 2 4 4 \
  --gpu 0

# 提示: 如果没有 Target 图片，把 --target_image 参数删掉即可