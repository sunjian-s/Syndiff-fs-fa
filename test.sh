#!/bin/bash

# ================= 配置区域 (请根据你的环境修改这里) =================

# 1. 你的数据文件夹路径 (请确保里面有 test 文件夹)
# 例如: /home/user/data/IXI 或者 ./dataset/IXI
INPUT_PATH="/mnt/mydata/gww/sj/syndiff/SynDiff/test_set/fundus" 

# 2. 结果保存路径
OUTPUT_PATH="./results"

# 3. 实验名称 (对应你训练时的 --exp 参数，脚本会去这里找模型)
EXP_NAME="exp_fundus_ffa_mask"

# 4. 你想使用的模型 Epoch (刚才看的日志里 76 效果不错)
EPOCH=80

# 5. 指定 GPU ID (如果你只有一张卡，就填 0)
GPU_ID=0

# 6. 模态设置 (T1 转 T2，或者 T2 转 T1)
CONTRAST1="T1"
CONTRAST2="T2"

# ===================================================================

echo "--------------------------------------------------------"
echo "开始运行测试 (Inference)..."
echo "使用模型: $EXP_NAME (Epoch $EPOCH)"
echo "数据路径: $INPUT_PATH"
echo "--------------------------------------------------------"

# 运行 Python 脚本
python test.py \
    --input_path "$INPUT_PATH" \
    --output_path "$OUTPUT_PATH" \
    --exp "$EXP_NAME" \
    --which_epoch $EPOCH \
    --contrast1 "$CONTRAST1" \
    --contrast2 "$CONTRAST2" \
    --gpu_chose $GPU_ID \
    --num_timesteps 4 \
    --ch_mult 1 1 2 2 4 4\
    --batch_size 4

# 注意：如果不希望计算 FID 可以保持默认；如果想算 FID，需要在上面加 --compute_fid

if [ $? -eq 0 ]; then
    echo "✅ 测试完成！"
    echo "结果已保存在: $OUTPUT_PATH/$EXP_NAME/generated_samples/epoch_$EPOCH"
else
    echo "❌ 运行出错，请检查上方的错误信息。"
fi