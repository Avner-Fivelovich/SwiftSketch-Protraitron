# SwiftSketch 96-Stroke Training Plan (Face-Oriented)

This document outlines the complete end-to-end pipeline required to train the SwiftSketch model for 96 strokes, specifically oriented towards faces by mixing the original dataset with FFHQ data. It also serves as a persistent project memory to aid in future development sessions.

---

## 🧠 Background Knowledge for Future Conversations

To quickly catch up in future sessions, here is the core context of the Portraitron SwiftSketch integration:
- **Project Goal:** Train a custom SwiftSketch diffusion model optimized for generating 96-stroke vector portraits. This involves mixing the original SwiftSketch data with a new dataset of faces (FFHQ).
- **The SwiftSketch Pipeline:** SwiftSketch is a Conditional Diffusion Sequence-to-Sequence Transformer that generates SVG strokes (4 bezier points per stroke). It takes visual conditioning via CLIP embeddings (`CLIPMiddle_layer4`).
- **Data Dependency (ControlSketch):** SwiftSketch requires "ground truth" vectorized targets (`svg_96s`) for training. These targets are created by running the highly expensive `ControlSketch/object_sketching.py` rasterizer over base images.
- **Data Format:** The repository operates on `.npz` dictionaries. Each dictionary starts with raw `image` data, is populated with `mask` and `attn_map` (via SDXL or auto-generation), gets a ground truth sketch (`svg_96s`) via ControlSketch, and finally receives a `CLIPMiddle_layer4_features` embedding.
- **Cluster Execution:** Because `ControlSketch` dataset generation is extremely slow, it is parallelized across GPU nodes on the TAU Slurm cluster using `slurm/generate_generation_jobs.py`.

---

## ✅ Finished Tasks

- **Dataset Generator Refactoring:** Modified `slurm/generate_generation_jobs.py` to accept dynamic command-line arguments (`--input_dir`, `--output_base_dir`, `--strokes`), allowing it to target either the Original dataset or the FFHQ dataset without hardcoding paths.
- **Training Script Patching:** Updated `SwiftSketch/train/train_SwiftSketch.py` to accept multiple directories in the `--train_data_dir` argument without overriding them, enabling the mixing of FFHQ and Original data.
- **Custom Slurm Training Script:** Created `slurm/run_train_custom_96s.slurm` to launch the 96-stroke training run on the cluster GPUs.
- **FFHQ .npz Validation:** Successfully wrote a script (`test_ffhq_npz.py`) that streamed an image from the Hugging Face `marcosv/ffhq-dataset` and packed it directly into the `.npz` format expected by ControlSketch, proving we don't need pre-computed SDXL attention maps if we supply the raw bytes.

---

## 🚧 Current & Upcoming Tasks

1. **Mass-download FFHQ to .npz:** Run `python download_ffhq_batch.py` to stream exactly 5,000 FFHQ images and convert them into base `.npz` dictionaries inside `data/ffhq_raw_npz`.
2. **Submit ControlSketch Slurm Jobs:** Run the `generate_generation_jobs.py` for both the Original dataset and the new FFHQ `.npz` dataset, then submit them to the queue using `submit_all_generation_jobs.sh` to bake in the `svg_96s` targets.
3. **Extract CLIP Features:** Run the `utils.get_features` script on the resulting 96-stroke directories.
4. **Train Base Diffusion Model:** Execute `slurm/run_train_custom_96s.slurm` pointing at both generated datasets to train the initial diffusion model.
5. **Train Refinement Network:** Generate intermediate sketches using the base model, then run `train_refine_model.py` to finalize the pipeline.

---

## 🛠 Phase 1: Data Preparation Pipeline

### 1. Format FFHQ Images to `.npz`
First, convert your raw FFHQ photos into the base `.npz` dictionaries required by the pipeline. Each `.npz` file must contain at least the `image` bytes (JPEG format) and optionally a `caption`. You can use your dataset generation scripts (`make_sdxl_data.py` or equivalent) to achieve this.

### 2. Generate 96-Stroke Targets (via ControlSketch)
SwiftSketch learns to draw by mimicking `ControlSketch`. You must generate the "ground truth" 96-stroke vector sketches (`svg_96s`) for both datasets. Use the custom batch scripts located in `slurm/` for this parallelized task:

**For Original Data:**
```bash
python slurm/generate_generation_jobs.py --input_dir "ControlSketch/data/train" --output_base_dir "data/original" --strokes 96
./slurm/submit_all_generation_jobs.sh 96
```

**For FFHQ Data:**
```bash
python slurm/generate_generation_jobs.py --input_dir "data/ffhq_raw_npz" --output_base_dir "data/ffhq" --strokes 96
./slurm/submit_all_generation_jobs.sh 96
```

### 3. Extract CLIP Image Features
SwiftSketch's conditioning mechanism relies on pre-extracted CLIP visual features (`CLIPMiddle_layer4`). Once the 96-stroke SVGs are baked into your `.npz` files (after the SLURM jobs finish), you must run this extraction tool on both directories:

```bash
cd SwiftSketch
python -m utils.get_features --dir_name "../data/original/strokes_96"
python -m utils.get_features --dir_name "../data/ffhq/strokes_96"
```

---

## 🚀 Phase 2: Training the Models

### 4. Train the Base Diffusion Model
Once features are extracted, you can train the main SwiftSketch generator. Point it to both the original and FFHQ directories so it learns the mixed distribution. You can use the `run_train_custom_96s.slurm` script, which essentially executes:

```bash
cd SwiftSketch
python -m train.train_SwiftSketch \
    --save_dir "./save/SwiftSketch_96s_FaceOriented" \
    --train_data_dir "../data/original/strokes_96" "../data/ffhq/strokes_96" \
    --num_strokes 96 \
    --cat_data_size 5000 \
    --batch_size 32 \
    --num_steps 60000 \
    --lr 5e-05 \
    --save_interval 10000
```
*(Checkpoints will be saved every 10,000 steps. The model will balance the batch with 5,000 Original objects and 5,000 FFHQ faces).*

### 5. Generate Intermediate Sketches for Refinement
SwiftSketch utilizes a two-stage process. To train the second stage (Refinement Network), you must use your newly trained base model to generate raw diffusion predictions (`svg_diffusion`) and bake them back into your dataset dictionaries:

```bash
cd SwiftSketch
python -m generate \
  --model_path "./save/SwiftSketch_96s_FaceOriented/model000600000.pt" \
  --use_refine 0 \
  --save_diffusion_sketch_in_dict 1 \
  --input_data "../data/original/strokes_96"

python -m generate \
  --model_path "./save/SwiftSketch_96s_FaceOriented/model000600000.pt" \
  --use_refine 0 \
  --save_diffusion_sketch_in_dict 1 \
  --input_data "../data/ffhq/strokes_96"
```

### 6. Train the Refinement Network
Finally, train the refinement network to clean up and polish the outputs from your base model:

```bash
cd SwiftSketch
python -m refine_model.train_refine.train_refine_model \
    --save_dir "./save/SwiftSketch_96s_FaceOriented_Refinement" \
    --resume_checkpoint "./save/SwiftSketch_96s_FaceOriented/model000600000.pt" \
    --train_data_dir "../data/original/strokes_96" "../data/ffhq/strokes_96" \
    --target_key_name "svg_96s" \
    --num_steps 60000 \
    --batch_size 32
```
