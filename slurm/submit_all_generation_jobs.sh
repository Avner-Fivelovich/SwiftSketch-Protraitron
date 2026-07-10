#!/bin/bash

# Configuration: Where to look for slurm files
SEARCH_DIR="./slurm/jobs"
STROKE_COUNTS=(48 64 96 128)

# Check if the directory exists
if [ ! -d "$SEARCH_DIR" ]; then
    echo "Error: Directory $SEARCH_DIR not found. Run python slurm/generate_generation_jobs.py first."
    exit 1
fi

echo "Searching and submitting jobs sequentially by stroke count..."

# Ensure the logs base directory exists
mkdir -p outputs/logs

for stroke in "${STROKE_COUNTS[@]}"; do
    SUB_DIR="$SEARCH_DIR/strokes_$stroke"
    if [ -d "$SUB_DIR" ]; then
        echo "==========================================="
        echo "Submitting jobs for $stroke strokes..."
        echo "==========================================="
        
        # Find all .slurm files in this specific subdirectory and submit them
        find "$SUB_DIR" -type f -name "*.slurm" | while read -r slurm_file; do
            echo "Submitting: $slurm_file"
            sbatch "$slurm_file"
            sleep 0.1  # Avoid overwhelming the scheduler
        done
        sleep 1 # Short pause between stroke groups
    else
        echo "Warning: Directory $SUB_DIR not found, skipping."
    fi
done

echo "Done. All generation jobs submitted to the queue."
echo "Use 'squeue --me' to check your job status."
squeue --me
