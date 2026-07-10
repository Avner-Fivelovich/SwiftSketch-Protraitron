import os
import sys

# Configuration
STROKE_COUNTS = [48, 64, 96, 128]
IMAGES_PER_JOB = 10
INPUT_DIR = "ControlSketch/data/train"
OUTPUT_BASE_DIR = "data"
BASE_SLURM_DIR = "slurm/jobs"
BASE_LOG_DIR = "outputs/logs"
PROJECT_DIR = "/vol/joberant_nobck/data/NLP_368307701_2526a/avnerf/SwiftSketch-Protraitron"

SLURM_TEMPLATE = """#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output={log_subdir}/{job_name}_%j.out
#SBATCH --error={log_subdir}/{job_name}_%j.err
#SBATCH --partition=studentkillable
#SBATCH --account=gpu-students
#SBATCH --constraint="RTX3090|RTX2080Ti|A5000|A6000|L40S"
#SBATCH --time=1440
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32000
#SBATCH --gpus=1

# 1. Activate environment
source ~/.bashrc

# Fallback for environment activation if .bashrc sourcing fails
if ! command -v conda &> /dev/null; then
    source /vol/joberant_nobck/data/NLP_368307701_2526a/$USER/anaconda3/bin/activate
fi

# Set HF cache directory, CLIP cache directory, and PyTorch CUDA allocator settings
export HF_HOME="/vol/joberant_nobck/data/NLP_368307701_2526a/$USER/huggingface_cache"
export CLIP_CACHE_DIR="/vol/joberant_nobck/data/NLP_368307701_2526a/$USER/clip_cache"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

conda activate swiftsketch_env

# 2. Ensure log directory exists and navigate to project directory
mkdir -p {log_subdir}
cd {project_dir}

# 3. Run Dataset Generation
python slurm/generate_dataset.py \\
  --num_strokes {num_strokes} \\
  --input_dir {input_dir} \\
  --output_dir {output_dir} \\
  --start_idx {start_idx} \\
  --limit {limit}
"""

def main():
    if not os.path.exists(INPUT_DIR):
        print(f"Error: Input directory {INPUT_DIR} does not exist.")
        sys.exit(1)
        
    # Count total .npz files in the input directory
    npz_files = []
    for root, _, files in os.walk(INPUT_DIR):
        for file in files:
            if file.endswith(".npz"):
                npz_files.append(file)
                
    total_files = len(npz_files)
    print(f"Found {total_files} total .npz files in {INPUT_DIR}.")
    
    # Calculate number of batches needed
    num_batches = (total_files + IMAGES_PER_JOB - 1) // IMAGES_PER_JOB
    print(f"Splitting into {num_batches} batches of {IMAGES_PER_JOB} images each per stroke count.")
    
    total_jobs = len(STROKE_COUNTS) * num_batches
    print(f"Generating a total of {total_jobs} SLURM scripts...")

    for num_strokes in STROKE_COUNTS:
        # Organize scripts and outputs by stroke count
        log_subdir = f"{BASE_LOG_DIR}/strokes_{num_strokes}"
        slurm_subdir = f"{BASE_SLURM_DIR}/strokes_{num_strokes}"
        output_dir = f"{OUTPUT_BASE_DIR}/controlsketch_{num_strokes}/train"
        
        os.makedirs(slurm_subdir, exist_ok=True)
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * IMAGES_PER_JOB
            job_name = f"ss_gen_{num_strokes}_b{batch_idx}"
            
            slurm_content = SLURM_TEMPLATE.format(
                job_name=job_name,
                log_subdir=log_subdir,
                project_dir=PROJECT_DIR,
                num_strokes=num_strokes,
                input_dir=INPUT_DIR,
                output_dir=output_dir,
                start_idx=start_idx,
                limit=IMAGES_PER_JOB,
            )
            
            file_path = f"{slurm_subdir}/{job_name}.slurm"
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(slurm_content)
                
    print(f"Done. Scripts generated in {BASE_SLURM_DIR}")

if __name__ == "__main__":
    main()
