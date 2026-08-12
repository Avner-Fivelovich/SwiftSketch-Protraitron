#!/bin/bash
#SBATCH --job-name=ss_train_96s
#SBATCH --output=outputs/logs/ss_train_96s_%j.out
#SBATCH --error=outputs/logs/ss_train_96s_%j.err
#SBATCH --partition=studentkillable
#SBATCH --account=gpu-students
#SBATCH --time=1440
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32000
#SBATCH --gpus=1

# 1. Activate the conda environment
source ~/.bashrc

if ! command -v conda &> /dev/null; then
    source /vol/joberant_nobck/data/NLP_368307701_2526a/$USER/anaconda3/bin/activate
fi

# Set HF cache directory, CLIP cache directory, and PyTorch CUDA allocator settings
export HF_HOME="/vol/joberant_nobck/data/NLP_368307701_2526a/$USER/huggingface_cache"
export CLIP_CACHE_DIR="/vol/joberant_nobck/data/NLP_368307701_2526a/$USER/clip_cache"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

conda activate swiftsketch_env

# 2. Ensure log and save directories exist
mkdir -p outputs/logs
mkdir -p outputs/train_96s_custom

cd "${SLURM_SUBMIT_DIR}"

# 3. Define the training dataset directories
# Update these paths to point to your specific original sectors and FFHQ generated datasets.
ORIGINAL_DATA="data/original/controlsketch_96/train"
FFHQ_DATA="data/ffhq/controlsketch_96/train" 

# 4. Execute the training script using multiple dataset directories
# Ensure SwiftSketch is in the python path
export PYTHONPATH=$PYTHONPATH:$(pwd)/SwiftSketch

python -m train.train_SwiftSketch \
    --num_strokes 96 \
    --train_data_dir "$ORIGINAL_DATA" "$FFHQ_DATA" \
    --cat_data_size 5000 \
    --save_dir "outputs/train_96s_custom" \
    --use_wandb 1 \
    --wandb_project_name "SwiftSketch-Protraitron" \
    --title "Train_96s_Custom_" \
    --batch_size 8 \
    --num_steps 120000 \
    --save_interval 20000
