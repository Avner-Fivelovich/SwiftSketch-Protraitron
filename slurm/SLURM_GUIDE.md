# 🚀 Running SwiftSketch / ControlSketch on the TAU CS Slurm Cluster

This guide explains how to set up the environment, compile dependencies with GPU support, sync the dataset, and execute optimization runs on the Slurm cluster.

Running on Slurm offers two major advantages:
1. **GPU-Accelerated Differentiable Rasterization**: Unlike macOS, the Slurm cluster's NVIDIA GPUs will run the differentiable rasterization (`pydiffvg`) on CUDA instead of the CPU. This removes the main CPU rasterization bottleneck.
2. **Fast Stable Diffusion Backpass**: SDS loss calculation will run on powerful cluster GPUs instead of Apple Silicon.

---

## 📋 Prerequisites
1. Ensure your **University VPN** is turned on.
2. Ensure you have SSH key access configured on the cluster.

---

## 🛠️ Step 1: Clone the Repo and Checkout the Branch on NetApp

Connect to the Slurm login node:
```bash
ssh avnerf@slurm-client.cs.tau.ac.il
```

> [!IMPORTANT]
> **Run Git Commands ONLY on the Login Node (`slurm-client.cs.tau.ac.il`)**:
> Cluster compute nodes (like `c-008`, `c-009`, etc.) run a minimal execution environment and do not have `git` installed (`git: Command not found`). 
> Always execute `git clone`, `git pull`, and other repo/branch management commands on the login node **before** starting interactive jobs or submitting batch scripts.

Navigate to your personal NetApp directory, clone the repository, and check out the main branch:
```bash
# Define your personal NetApp path
export MY_NETAPP_PATH="/vol/joberant_nobck/data/NLP_368307701_2526a/$USER"
cd $MY_NETAPP_PATH

# Clone the repository
git clone <URL_TO_REPOS_OR_PATH> SwiftSketch-Protraitron
cd SwiftSketch-Protraitron

# Fetch all branches and check out main (or your development branch)
git fetch origin
git checkout main
```

---

## 🐍 Step 2: Set up the Conda Environment

Create a new Conda environment (`swiftsketch_env`) using Python 3.9:
```bash
# Refresh terminal to ensure conda is available
source ~/.bashrc

# Configure package directory on NetApp to avoid home quota limit
conda config --add pkgs_dirs /vol/joberant_nobck/data/NLP_368307701_2526a/$USER/conda_pkgs

# Create the environment
conda create -y -n swiftsketch_env python=3.9.19

# Activate the environment
conda activate swiftsketch_env
```

---

## 📦 Step 3: Install Dependencies

With the environment activated, run the following commands to install PyTorch with CUDA 12.1 support and all required libraries:

```bash
# 1. Install PyTorch with CUDA 12.1
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121

# 2. Install requirements using the relaxed requirements file
pip install -r slurm/requirements_relaxed.txt

# 3. Install OpenAI CLIP
pip install git+https://github.com/openai/CLIP.git
```

### ⚡ Compile `diffvg` with CUDA support
To run the differentiable rasterizer on the GPU, you must compile `diffvg` on the cluster. Since CUDA is available, `setup.py` will automatically compile with CUDA support.

```bash
# 4. Clone and install diffvg
git clone --recursive https://github.com/BachiLi/diffvg.git
cd diffvg

# Run the installer (compilation takes 1-2 minutes)
python setup.py install

# Go back to the main directory
cd ..
```

---

## 💾 Step 4: Sync the Datasets
Since the datasets under `ControlSketch/data/` (such as `train/` and `test/` folders, or `.tar.gz` files) are local and not committed to git, you need to copy them from your local Mac to NetApp.

Run the following command **from your local Mac terminal** (ensure VPN is on):
```bash
rsync -avz --progress /Users/avnerf/Documents/GitHub/SwiftSketch-Protraitron/ControlSketch/data/ avnerf@slurm-client.cs.tau.ac.il:/vol/joberant_nobck/data/NLP_368307701_2526a/avnerf/SwiftSketch-Protraitron/ControlSketch/data/
```

---

## 🏃‍♂️ Step 5: Submit Jobs to Slurm

We have prepared two Slurm batch scripts for running the step comparison experiments:
* `run_step_comparison.slurm` (runs 16-stroke optimization)
* `run_step_comparison_64.slurm` (runs 64-stroke optimization)

> [!IMPORTANT]
> **Create the log directories before submitting**:
> Slurm requires the destination paths for stdout/stderr logs to exist prior to job launch. Create the directories from the repository root:
> ```bash
> mkdir -p outputs/logs
> ```

To submit either job, run:
```bash
# Submit the 16-stroke comparison job
sbatch slurm/run_step_comparison.slurm

# Submit the 64-stroke comparison job
sbatch slurm/run_step_comparison_64.slurm
```

### 📊 Useful Slurm Commands
* Check your active and pending jobs:
  ```bash
  squeue --me
  ```
* Stream/view the live output logs:
  ```bash
  tail -f outputs/logs/ss_comp_16_<JOB_ID>.out
  ```
* Cancel a running job:
  ```bash
  scancel <JOB_ID>
  ```

---

## 🎨 Step 6: Dataset Generation in Batches (Slurm Job Arrays)

To prepare the dataset for model training, you must process the raw `.npz` files through ControlSketch to generate the customized stroke vector keys (e.g., `svg_16s` or `svg_64s`). Since cluster jobs have a 24-hour execution limit, we split the processing into parallel batches of **10 pictures per job** using a Slurm Job Array.

### 🏃‍♂️ Running the Array Job
From the repository root directory on the login node, submit the job array. Specify the array range corresponding to your dataset size:
```bash
# Example: Process 300 pictures (30 jobs of 10 pictures each)
# Task indices will be 0, 1, 2, ..., 29.
# This limits execution to 5 concurrent jobs at any time (%5) to share cluster resources politely.
sbatch --array=0-29%5 slurm/run_dataset_generation.slurm
```

To run the generation for a different number of strokes (e.g., 64 strokes instead of the default 16), export the `NUM_STROKES` environment variable:
```bash
export NUM_STROKES=64
sbatch --array=0-29%5 slurm/run_dataset_generation.slurm
```

### 📂 Output Management
- Each parallel array task processes a slice of files starting at index `START_IDX = SLURM_ARRAY_TASK_ID * 10` up to `START_IDX + 9`.
- The outputs are saved directly to `data/controlsketch_<STROKES>/train/`. Since each job writes to unique filenames, they do not overlap.
- Temporary logs are created in unique directories `_temp_sketch_logs_<START_IDX>_<STROKES>` to avoid race conditions.
- Once all jobs complete, copy the fully accumulated `data/controlsketch_<STROKES>/` directory to copy the results for model training.
