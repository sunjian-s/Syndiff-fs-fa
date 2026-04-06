#!/bin/bash

# SynDiff root-dir inference templates aligned with train.sh ablations

ROOT=/mnt/mydata/gww/sj/syndiff/SynDiff
GPU=3

INPUT_IMG="/mnt/mydata/gww/sj/syndiff/SynDiff/test_set/fundus/25.jpg"
TARGET_IMG="/mnt/mydata/gww/sj/syndiff/SynDiff/test_set/ffa/25.jpg"

# 1) Recommended first check: cbam local 128 64
MODEL_PATH="${ROOT}/results/exp_-baseline+w+l+ms+vgg/gen_diffusive_2_300.pth"
OUTPUT_DIR="./final_results/exp_-baseline+w+l+ms+vgg"

cd ${ROOT} && CUDA_VISIBLE_DEVICES=${GPU} python inference2.py \
  --model_path "${MODEL_PATH}" \
  --input_image "${INPUT_IMG}" \
  --target_image "${TARGET_IMG}" \
  --output_path "${OUTPUT_DIR}" \
  --image_size 256 \
  --num_channels 3 \
  --num_channels_dae 64 \
  --num_timesteps 4 \
  --num_res_blocks 2 \
  --ch_mult 1 1 2 2 4 4 \
  --attn_resolutions 16 \
  --local_attn_type cbam \
  --local_attn_resolutions 128 64 32 \
  --gpu 0

# 2) No local attention
# MODEL_PATH="${ROOT}/results1880/exp_no_local_attn_resize512p10/gen_diffusive_2_100.pth"
# OUTPUT_DIR="./final_exp_no_local_attn_resize512p10"
#
# cd ${ROOT} && CUDA_VISIBLE_DEVICES=${GPU} python inference2.py \
#   --model_path "${MODEL_PATH}" \
#   --input_image "${INPUT_IMG}" \
#   --target_image "${TARGET_IMG}" \
#   --output_path "${OUTPUT_DIR}" \
#   --image_size 256 \
#   --num_channels 3 \
#   --num_channels_dae 64 \
#   --num_timesteps 4 \
#   --num_res_blocks 2 \
#   --ch_mult 1 1 2 2 4 4 \
#   --attn_resolutions 16 \
#   --local_attn_type none \
#   --gpu 0

# 3) Only one local level: 64
# MODEL_PATH="${ROOT}/results1880/exp_cbam_local_64_resize512p10/gen_diffusive_2_100.pth"
# OUTPUT_DIR="./final_exp_cbam_local_64_resize512p10"
#
# cd ${ROOT} && CUDA_VISIBLE_DEVICES=${GPU} python inference2.py \
#   --model_path "${MODEL_PATH}" \
#   --input_image "${INPUT_IMG}" \
#   --target_image "${TARGET_IMG}" \
#   --output_path "${OUTPUT_DIR}" \
#   --image_size 256 \
#   --num_channels 3 \
#   --num_channels_dae 64 \
#   --num_timesteps 4 \
#   --num_res_blocks 2 \
#   --ch_mult 1 1 2 2 4 4 \
#   --attn_resolutions 16 \
#   --local_attn_type cbam \
#   --local_attn_resolutions 64 \
#   --gpu 0

# 4) Control: same as 128 64, but lower cycle adversarial weight
# MODEL_PATH="${ROOT}/results1880/exp_cbam_local_128_64_cycleadv02_resize512p10/gen_diffusive_2_100.pth"
# OUTPUT_DIR="./final_exp_cbam_local_128_64_cycleadv02_resize512p10"
#
# cd ${ROOT} && CUDA_VISIBLE_DEVICES=${GPU} python inference2.py \
#   --model_path "${MODEL_PATH}" \
#   --input_image "${INPUT_IMG}" \
#   --target_image "${TARGET_IMG}" \
#   --output_path "${OUTPUT_DIR}" \
#   --image_size 256 \
#   --num_channels 3 \
#   --num_channels_dae 64 \
#   --num_timesteps 4 \
#   --num_res_blocks 2 \
#   --ch_mult 1 1 2 2 4 4 \
#   --attn_resolutions 16 \
#   --local_attn_type cbam \
#   --local_attn_resolutions 128 64 \
#   --gpu 0

# If you do not have target FFA for a case, remove: --target_image "${TARGET_IMG}"
