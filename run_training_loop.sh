#!/bin/bash

# Configuration
SAVE_DIR="outputs/train_96s_mac_f16"
SUB_DIR="outputs/train_96s_mac_f16/Train_96s_MacM4_MPS_AutoResumeCLIPMiddle_layer4_seed20_0.2lpips_1.0L1P"
RESUME_CKPT="outputs/train_96s_mac_f16/model000076000.pt"

while true; do
    echo "=========================================================="
    echo "Starting Training Run (Saving every 500 steps)"
    echo "=========================================================="
    
    # Run the training command
    PYTHONPATH=$PYTHONPATH:$(pwd)/SwiftSketch PYTHONUNBUFFERED=1 python -u -m train.train_SwiftSketch \
        --num_strokes 96 \
        --train_data_dir "data/original/controlsketch_96/train" \
        --cat_data_size 25000 \
        --cache_path_dir "outputs/train_96s_custom" \
        --save_dir "$SAVE_DIR" \
        --resume_checkpoint "$RESUME_CKPT" \
        --use_wandb 1 \
        --wandb_project_name "SwiftSketch-Protraitron" \
        --title "Train_96s_MacM4_MPS_AutoResume" \
        --batch_size 32 \
        --num_steps 200000 \
        --save_interval 500 \
        > >(tee outputs/logs/m4_train_resumed_mps.out) \
        2> >(tee outputs/logs/m4_train_resumed_mps.err >&2)
        
    echo "Training script exited (likely out of memory). Cleaning up checkpoints..."
    
    # Python script to identify the latest checkpoint and delete non-2000 intermediate checkpoints
    python3 -c "
import os
import re

save_dir = '$SUB_DIR'
if not os.path.exists(save_dir):
    exit()

files = os.listdir(save_dir)

# Find all valid model numbers
models = []
for f in files:
    match = re.match(r'model(\d+).pt$', f)
    if match:
        models.append(int(match.group(1)))

if not models:
    exit()

latest_step = max(models)
print(f'Latest step found: {latest_step}')

# Keep the latest, and any step divisible by 2000
for step in models:
    if step == latest_step or step % 2000 == 0:
        continue
    
    model_file = os.path.join(save_dir, f'model{step:09d}.pt')
    opt_file = os.path.join(save_dir, f'opt{step:09d}.pt')
    
    if os.path.exists(model_file):
        os.remove(model_file)
        print(f'Deleted intermediate: {model_file}')
    if os.path.exists(opt_file):
        os.remove(opt_file)
        print(f'Deleted intermediate: {opt_file}')
"
    
    echo "Cleanup complete. Restarting loop in 5 seconds..."
    sleep 5
done
