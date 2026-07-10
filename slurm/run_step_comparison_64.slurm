#!/bin/bash
#SBATCH --job-name=ss_comp_64
#SBATCH --output=outputs/logs/ss_comp_64_%j.out
#SBATCH --error=outputs/logs/ss_comp_64_%j.err
#SBATCH --partition=studentkillable
#SBATCH --account=gpu-students
#SBATCH --time=1440
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32000
#SBATCH --gpus=1

# 1. Activate the conda environment
source ~/.bashrc

# Set HF cache directory to NetApp to avoid home directory quota errors
export HF_HOME="/vol/joberant_nobck/data/NLP_368307701_2526a/$USER/huggingface_cache"

# Activate the conda environment for swiftsketch
conda activate swiftsketch_env

# 2. Navigate to the project directory from which sbatch was executed
cd "${SLURM_SUBMIT_DIR}"

# 3. Execute the 64-stroke step comparison script
python slurm/run_step_comparison_64.py
