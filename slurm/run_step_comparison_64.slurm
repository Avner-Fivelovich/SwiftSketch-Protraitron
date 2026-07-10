#!/bin/bash
#SBATCH --job-name=ss_comp_64
#SBATCH --output=outputs/logs/ss_comp_64_%j.out
#SBATCH --error=outputs/logs/ss_comp_64_%j.err
#SBATCH --partition=killable
#SBATCH --account=gpu-students
#SBATCH --time=1440
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32000
#SBATCH --gpus=1

# 1. Activate the conda environment
source ~/.bashrc

# Fallback for environment activation if .bashrc sourcing fails
if ! command -v conda &> /dev/null; then
    source /vol/joberant_nobck/data/NLP_368307701_2526a/$USER/anaconda3/bin/activate
fi

# Set HF cache directory, CLIP cache directory, and PyTorch CUDA allocator settings
export HF_HOME="/vol/joberant_nobck/data/NLP_368307701_2526a/$USER/huggingface_cache"
export CLIP_CACHE_DIR="/vol/joberant_nobck/data/NLP_368307701_2526a/$USER/clip_cache"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Activate the conda environment for swiftsketch
conda activate swiftsketch_env

# 2. Ensure log directories exist and navigate to the project directory from which sbatch was executed
mkdir -p outputs/logs
cd "${SLURM_SUBMIT_DIR}"

# 3. Execute the 64-stroke step comparison script
python slurm/run_step_comparison_64.py
