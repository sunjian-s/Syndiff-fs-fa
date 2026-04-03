# cd /mnt/mydata/gww/sj/syndiff/SynDiff && CUDA_VISIBLE_DEVICES=3 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python3 train.py \
# --image_size 256 --exp exp_-baseline+w+l+cbam --num_channels 3 --input_path ./SynDiff_dataset_with_mask --output_path ./results \
# --num_channels_dae 64 --ch_mult 1 1 2 2 4 4 --num_timesteps 10 --num_res_blocks 2 --batch_size 4 --num_epoch 300 --contrast1 contrast1 \
# --contrast2 contrast2 --save_content --save_content_every 1 --lambda_l1_loss 2.0 --lazy_reg 10 --r1_gamma 1.0 --attn_resolutions 16 8 \
# --local_attn_type cbam --local_attn_resolutions 256 128 64 --port_num 23463
# cd /mnt/mydata/gww/sj/syndiff/SynDiff && CUDA_VISIBLE_DEVICES=3 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python3 train.py \
# --image_size 256 --exp exp_-baseline+w+l+scsa --num_channels 3 --input_path ./SynDiff_dataset_with_mask --output_path ./results \
# --num_channels_dae 64 --ch_mult 1 1 2 2 4 4 --num_timesteps 4 --num_res_blocks 2 --batch_size 4 --num_epoch 200 --contrast1 contrast1 \
# --contrast2 contrast2 --save_content --save_content_every 1 --lambda_l1_loss 2.0 --lazy_reg 10 --r1_gamma 1.0 --port_num 23464 \
# --attn_resolutions 16 8 \
# --local_attn_type scsa --local_attn_resolutions 256 128 64 --resume
# cd /mnt/mydata/gww/sj/syndiff/SynDiff && CUDA_VISIBLE_DEVICES=3 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python3 train.py \
# --image_size 256 --exp exp_-baseline+w+l+ms+vgg003 --num_channels 3 --input_path ./SynDiff_dataset --output_path ./results1880 \
# --ch_mult 1 1 2 2 4 4 --num_timesteps 4 --num_res_blocks 2 --batch_size 4 --num_epoch 200 --contrast1 contrast1 \
# --contrast2 contrast2 --save_content --save_content_every 1 --lambda_l1_loss 0.5 --lazy_reg 10 --r1_gamma 1.0 --attn_resolutions 16 8 \
# --local_attn_type cbam --local_attn_resolutions 256 128 64 --port_num 23463  \
cd /mnt/mydata/gww/sj/syndiff/SynDiff && CUDA_VISIBLE_DEVICES=3 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python3 train.py \
--image_size 256 --exp exp_baseline_pure_syndiff+cbam_local --num_channels 3 --input_path ./SynDiff_dataset --output_path ./results1880 \
--ch_mult 1 1 2 2 4 4 --num_timesteps 4 --num_res_blocks 2 --batch_size 4 --num_epoch 200 --contrast1 contrast1 \
--contrast2 contrast2 --save_content --save_content_every 20 \
--lambda_l1_loss 0.5 \
--lazy_reg 10 --r1_gamma 1.0 \
--attn_resolutions 16 \
--local_attn_type cbam --local_attn_resolutions 256 128 64 \
--port_num 23463






 
