#!/bin/bash
#SBATCH --job-name=ss_gen_dataset
#SBATCH --output=outputs/logs/ss_gen_%A_%a.out
#SBATCH --error=outputs/logs/ss_gen_%A_%a.err
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

# 3. Calculate start index for this array task
# Since each task processes 10 pictures, start_idx is task_id * 10
START_IDX=$((SLURM_ARRAY_TASK_ID * 10))
STROKE_COUNT=${NUM_STROKES:-16}

echo "=========================================================="
echo "Starting Slurm Array Job Task"
echo "Job ID: ${SLURM_ARRAY_JOB_ID} | Array Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Processing images ${START_IDX} to $((START_IDX + 9))"
echo "Strokes count: ${STROKE_COUNT}"
echo "=========================================================="

# 4. Execute the dataset generation script for 10 pictures
python slurm/generate_dataset.py \
  --num_strokes ${STROKE_COUNT} \
  --input_dir ControlSketch/data/train \
  --output_dir data/controlsketch_${STROKE_COUNT}/train \
  --start_idx ${START_IDX} \
  --limit 10
