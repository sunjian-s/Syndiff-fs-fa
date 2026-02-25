#!/bin/bash

# ================= 配置区域 =================
# 1. 模型路径
MODEL_PATH="/mnt/mydata/gww/sj/syndiff/SynDiff/results/exp_-baseline+w+l+scsa/gen_diffusive_2_200.pth"

# 2. 原始大图路径
INPUT_IMG="/mnt/mydata/gww/sj/syndiff/SynDiff/test_set/fundus/27.jpg"

# 3. 真值大图路径
TARGET_IMG="/mnt/mydata/gww/sj/syndiff/SynDiff/test_set/ffa/27.jpg"

# 4. 输出目录
OUTPUT_DIR="./final_-baseline+w+l+scsa"
# ===========================================

export CUDA_VISIBLE_DEVICES=3

# 关键修正：添加了 --num_channels_dae, --attn_resolutions 以及所有 SCSA 相关参数
python inference2.py \
  --model_path "$MODEL_PATH" \
  --input_image "$INPUT_IMG" \
  --target_image "$TARGET_IMG" \
  --output_path "$OUTPUT_DIR" \
  --image_size 256 \
  --num_channels 3 \
  --num_channels_dae 64 \
  --ch_mult 1 1 2 2 4 4 \
  --num_timesteps 4 \
  --num_res_blocks 2 \
  --attn_resolutions 16 8 \
  --local_attn_type scsa \
  --local_attn_resolutions 256 128 64 \
  --gpu 0