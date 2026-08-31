#!/bin/bash
export PYTHONPATH=$PYTHONPATH:$(pwd)/SwiftSketch
python -m train.train_SwiftSketch \
    --num_strokes 96 \
    --train_data_dir "data/ffhq/controlsketch_96/train" \
    --cat_data_size 10 \
    --save_dir "outputs/train_96s_custom2/Train_96s_MacM4_CLIPMiddle_layer4_seed20_0.2lpips_1.0L1P" \
    --use_wandb 0 \
    --use_data_cache 1 \
    --cache_path_dir "outputs/train_96s_custom" \
    --batch_size 2 \
    --num_steps 2 \
    --save_interval 1
