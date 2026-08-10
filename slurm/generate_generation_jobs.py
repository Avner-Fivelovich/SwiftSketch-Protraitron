import os
import sys
import argparse

# Default Configuration
IMAGES_PER_JOB = 100
PROJECT_DIR = "/vol/joberant_nobck/data/NLP_368307701_2526a/avnerf/SwiftSketch-Protraitron"
BASE_SLURM_DIR = "slurm/jobs"
BASE_LOG_DIR = "outputs/logs"

def parse_args():
    parser = argparse.ArgumentParser(description="Generate SLURM batch scripts for dataset generation.")
    parser.add_argument("--input_dir", type=str, default="ControlSketch/data/train", help="Directory containing source .npz files")
    parser.add_argument("--output_base_dir", type=str, default="data", help="Base output directory")
    parser.add_argument("--strokes", type=int, nargs="+", default=[48, 64, 96, 128], help="List of stroke counts")
    parser.add_argument("--images_per_job", type=int, default=100, help="Images processed per slurm job")
    parser.add_argument("--max_files", type=int, default=None, help="Maximum number of files to process overall (useful for limiting huge datasets)")
    parser.add_argument("--allowed_categories", type=str, nargs="+", default=None, help="List of specific subdirectories to include")
    parser.add_argument("--job_prefix", type=str, default="orig", help="Prefix for the generated job names (e.g. orig or ffhq)")
    parser.add_argument("--specific_batches", type=int, nargs="+", default=None, help="Optional specific batch indices to generate (e.g., 21 47). If provided, only these batches are generated.")
    return parser.parse_args()

SLURM_TEMPLATE = """#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output={log_subdir}/{job_name}_%j.out
#SBATCH --error={log_subdir}/{job_name}_%j.err
#SBATCH --partition=studentkillable
#SBATCH --account=gpu-students
#SBATCH --time=1440
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=15000
#SBATCH --gpus=1
#SBATCH --prefer="geforce_rtx_2080"

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
  --limit {limit} {max_files_flag} {allowed_cats_flag}
"""

def main():
    args = parse_args()
    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory {args.input_dir} does not exist.")
        sys.exit(1)
        
    # Group .npz files by category (subdirectory)
    from collections import defaultdict
    category_files = defaultdict(list)
    
    for root, _, files in os.walk(args.input_dir):
        for file in files:
            if file.endswith(".npz"):
                # Use the relative directory as the category
                rel_dir = os.path.relpath(root, args.input_dir)
                if args.allowed_categories is not None and rel_dir not in args.allowed_categories:
                    continue
                category_files[rel_dir].append(os.path.join(root, file))
                
    npz_files = []
    if args.max_files is not None:
        # Calculate how many files to take from each category for a perfectly balanced subset
        num_categories = len(category_files)
        files_per_category = args.max_files // num_categories
        remainder = args.max_files % num_categories
        
        print(f"Stratifying {args.max_files} files across {num_categories} categories...")
        # Sort categories to ensure 100% deterministic ordering across cluster nodes!
        for i, cat in enumerate(sorted(category_files.keys())):
            files = category_files[cat]
            # Distribute the remainder across the first few categories
            take_count = files_per_category + (1 if i < remainder else 0)
            
            # Sort for deterministic picking
            files.sort()
            npz_files.extend(files[:take_count])
            
        print(f"Balanced selection: ~{files_per_category} files per category.")
    else:
        for cat in sorted(category_files.keys()):
            files = category_files[cat]
            files.sort()
            npz_files.extend(files)
            
    total_files = len(npz_files)
    print(f"Found {total_files} total .npz files to process.")
    
    # Calculate number of batches needed
    num_batches = (total_files + args.images_per_job - 1) // args.images_per_job
    print(f"Splitting into {num_batches} batches of {args.images_per_job} images each per stroke count.")
    
    total_jobs = len(args.strokes) * num_batches
    print(f"Generating a total of {total_jobs} SLURM scripts...")

    import shutil
    for num_strokes in args.strokes:
        # Organize scripts and outputs by stroke count
        log_subdir = f"{BASE_LOG_DIR}/strokes_{num_strokes}"
        slurm_subdir = f"{BASE_SLURM_DIR}/strokes_{num_strokes}"
        output_dir = f"{args.output_base_dir}/controlsketch_{num_strokes}/train"
        
        if os.path.exists(slurm_subdir):
            shutil.rmtree(slurm_subdir)
        os.makedirs(slurm_subdir, exist_ok=True)
        
        for batch_idx in range(num_batches):
            if args.specific_batches is not None and batch_idx not in args.specific_batches:
                continue
                
            start_idx = batch_idx * args.images_per_job
            job_name = f"{args.job_prefix}B{batch_idx}_{num_strokes}_ss"
            
            
            max_files_flag = f"--max_files {args.max_files}" if args.max_files is not None else ""
            allowed_cats_flag = f"--allowed_categories {' '.join(args.allowed_categories)}" if args.allowed_categories is not None else ""
            
            slurm_content = SLURM_TEMPLATE.format(
                job_name=job_name,
                log_subdir=log_subdir,
                project_dir=PROJECT_DIR,
                num_strokes=num_strokes,
                input_dir=args.input_dir,
                output_dir=output_dir,
                start_idx=start_idx,
                limit=args.images_per_job,
                max_files_flag=max_files_flag,
                allowed_cats_flag=allowed_cats_flag
            )
            
            file_path = f"{slurm_subdir}/{job_name}.slurm"
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(slurm_content)
                
    print(f"Done. Scripts generated in {BASE_SLURM_DIR}")

if __name__ == "__main__":
    main()
