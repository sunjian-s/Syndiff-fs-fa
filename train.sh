#!/bin/bash

# SynDiff root-dir ablations
# Current priority:
# 1. local_attn_resolutions = 128 64
# 2. local_attn_type = none
# 3. local_attn_resolutions = 64
# 4. control: lower lambda_cycle_adv

ROOT=/mnt/mydata/gww/sj/syndiff/SynDiff
DATASET=./SynDiff_dataset_with_mask
OUT=./results
GPU=3

# 1) Recommended first run: cbam local at 128 64
cd ${ROOT} && CUDA_VISIBLE_DEVICES=${GPU} PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python3 train.py \
--image_size 256 --exp exp_cbam_local_128_64-vgg --num_channels 3 --input_path ${DATASET} --output_path ${OUT} \
--ch_mult 1 1 2 2 4 4 --num_timesteps 4 --num_res_blocks 2 --batch_size 4 --num_epoch 200 --contrast1 contrast1 \
--contrast2 contrast2 --save_content --save_content_every 20 \
--lambda_pair_l1 0.05 --lambda_cycle_l1 0.05 --lambda_cycle_adv 0.2 \
--lambda_lumi 0.2 --lambda_wavelet 0.1 --lambda_vgg 0.1 \
--lumi_bright_factor 3.0 --lumi_dark_factor 1.0 --lumi_focus_sharpness 2.0 \
--wavelet_high_weight 1.5 \
--lazy_reg 10 --r1_gamma 1.0 \
--attn_resolutions 16 \
--local_attn_type cbam --local_attn_resolutions 128 64 --not_use_tanh \
--port_num 23464

# 2) No local attention
# cd ${ROOT} && CUDA_VISIBLE_DEVICES=${GPU} PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python3 train.py \
# --image_size 256 --exp exp_no_local_attn_resize512p10 --num_channels 3 --input_path ${DATASET} --output_path ${OUT} \
# --ch_mult 1 1 2 2 4 4 --num_timesteps 4 --num_res_blocks 2 --batch_size 4 --num_epoch 200 --contrast1 contrast1 \
# --contrast2 contrast2 --save_content --save_content_every 20 \
# --lambda_pair_l1 0.5 --lambda_cycle_l1 0.25 --lambda_cycle_adv 0.2 \
# --lambda_lumi 0.2 --lambda_wavelet 0.1 \
# --lumi_bright_factor 3.0 --lumi_dark_factor 1.0 --lumi_focus_sharpness 2.0 \
# --wavelet_high_weight 1.5 \
# --lazy_reg 10 --r1_gamma 1.0 \
# --attn_resolutions 16 \
# --local_attn_type none \
# --port_num 23464

# 3) Only one local level: 64
# cd ${ROOT} && CUDA_VISIBLE_DEVICES=${GPU} PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python3 train.py \
# --image_size 256 --exp exp_cbam_local_64_resize512p10 --num_channels 3 --input_path ${DATASET} --output_path ${OUT} \
# --ch_mult 1 1 2 2 4 4 --num_timesteps 4 --num_res_blocks 2 --batch_size 4 --num_epoch 200 --contrast1 contrast1 \
# --contrast2 contrast2 --save_content --save_content_every 20 \
# --lambda_pair_l1 0.5 --lambda_cycle_l1 0.25 --lambda_cycle_adv 0.2 \
# --lambda_lumi 0.2 --lambda_wavelet 0.1 \
# --lumi_bright_factor 3.0 --lumi_dark_factor 1.0 --lumi_focus_sharpness 2.0 \
# --wavelet_high_weight 1.5 \
# --lazy_reg 10 --r1_gamma 1.0 \
# --attn_resolutions 16 \
# --local_attn_type cbam --local_attn_resolutions 64 \
# --port_num 23465

# 4) Control: same as 128 64, but lower cycle adversarial weight
# cd ${ROOT} && CUDA_VISIBLE_DEVICES=${GPU} PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python3 train.py \
# --image_size 256 --exp exp_cbam_local_128_64_cycleadv02_resize512p10 --num_channels 3 --input_path ${DATASET} --output_path ${OUT} \
# --ch_mult 1 1 2 2 4 4 --num_timesteps 4 --num_res_blocks 2 --batch_size 4 --num_epoch 200 --contrast1 contrast1 \
# --contrast2 contrast2 --save_content --save_content_every 20 \
# --lambda_pair_l1 0.5 --lambda_cycle_l1 0.25 --lambda_cycle_adv 0.2 \
# --lambda_lumi 0.2 --lambda_wavelet 0.1 \
# --lumi_bright_factor 3.0 --lumi_dark_factor 1.0 --lumi_focus_sharpness 2.0 \
# --wavelet_high_weight 1.5 \
# --lazy_reg 10 --r1_gamma 1.0 \
# --attn_resolutions 16 \
# --local_attn_type cbam --local_attn_resolutions 128 64 \
# --port_num 23466
