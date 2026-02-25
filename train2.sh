cd /mnt/mydata/gww/sj/syndiff/SynDiff && \
CUDA_VISIBLE_DEVICES=3 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python3 train.py \
--image_size 256 \
--exp exp_fundus_ffa_mask1 \
--num_channels 3 \
--input_path ./SynDiff_dataset_with_mask \
--output_path ./results \
--num_channels_dae 64 \
--ch_mult 1 1 2 2 4 4 \
--num_timesteps 4 \
--num_res_blocks 2 \
--batch_size 4 \
--num_epoch 301 \
--contrast1 contrast1 \
--contrast2 contrast2 \
--save_content \
--save_content_every 1 \
--lazy_reg 10 \
--r1_gamma 1.0 \
--port_num 23457 \
--lambda_l1_loss 20.0 \
--resume \
--lr_g 0.00005 \
--lr_d 0.00005