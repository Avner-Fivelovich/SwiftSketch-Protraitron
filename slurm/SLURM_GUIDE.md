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
git clone https://github.com/Avner-Fivelovich/SwiftSketch-Protraitron.git SwiftSketch-Protraitron
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

To bypass disk quota limits (`Disk quota exceeded` errors) on the home folder, we redirect all caches and temp builds to NetApp. Run this in your terminal:

```bash
# 1. Create and redirect cache/temp folders to NetApp
mkdir -p /vol/joberant_nobck/data/NLP_368307701_2526a/$USER/{pip_cache,tmp}
export PIP_CACHE_DIR="/vol/joberant_nobck/data/NLP_368307701_2526a/$USER/pip_cache"
export TMPDIR="/vol/joberant_nobck/data/NLP_368307701_2526a/$USER/tmp"

# 2. Force install PyTorch with CUDA 12.1 compatibility
# Note: Pinned to 2.3.1 to support sm_61 (Titan XP) architectures on the student partition.
pip install --force-reinstall torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121

# 3. Install requirements using the relaxed requirements file (flash-attn is skipped)
pip install -r slurm/requirements_relaxed.txt

# 4. Install OpenAI CLIP
pip install git+https://github.com/openai/CLIP.git
```

### ⚡ Compile `diffvg` with CUDA support
To run the differentiable rasterizer on the GPU, you must compile `diffvg` on the cluster. Since CUDA is available, `setup.py` will automatically compile with CUDA support. Modern CMake versions require a policy patch to build successfully:

```bash
# 5. Clone diffvg
git clone --recursive https://github.com/BachiLi/diffvg.git
cd diffvg

# 6. Apply policy patch to setup.py to compile successfully
python -c "
with open('setup.py', 'r') as f:
    code = f.read()
code = code.replace(\"cmake_args = [\", \"cmake_args = ['-DCMAKE_POLICY_VERSION_MINIMUM=3.5', '-DCMAKE_CXX_STANDARD=14', \")
with open('setup.py', 'w') as f:
    f.write(code)
"

# 7. Run the installer (compilation takes 1-2 minutes)
python setup.py install

# 8. Go back to the main directory and clean up
cd ..
rm -rf diffvg/
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

## 🎨 Step 6: Dataset Generation in Batches (Multi-Job Scripts)

To prepare the dataset for model training, you must process the raw `.npz` files through ControlSketch to generate the customized stroke vector keys (e.g., `svg_48s`, `svg_64s`, `svg_96s`, or `svg_128s`). Rather than a single job array, we use a Python script generator and a bash submission script to queue jobs sequentially by stroke count.

### 🏃‍♂️ Running the Jobs
1. **Generate the Slurm scripts**:
   This script scans the input folder, divides the images into batches of 10, and writes statically configured `.slurm` scripts for `48`, `64`, `96`, and `128` strokes under `slurm/jobs/`:
   ```bash
   python slurm/generate_generation_jobs.py
   ```

2. **Submit the jobs to the queue**:
   This bash script submits the generated scripts sequentially (all 48-stroke jobs first, followed by 64-stroke, 96-stroke, and finally 128-stroke jobs) with a short 0.1-second pause to be polite to the scheduler:
   ```bash
   ./slurm/submit_all_generation_jobs.sh
   ```

### 📂 Output Management
* **Generated Dataset (Source of Truth)**:
  * `data/controlsketch_48/train/`
  * `data/controlsketch_64/train/`
  * `data/controlsketch_96/train/`
  * `data/controlsketch_128/train/`
* **Log Files (Stdout / Stderr)**:
  * `outputs/logs/strokes_48/`
  * `outputs/logs/strokes_64/`
  * `outputs/logs/strokes_96/`
  * `outputs/logs/strokes_128/`

