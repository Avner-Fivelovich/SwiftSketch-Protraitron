# SwiftSketch: A Diffusion Model for Image-to-Vector Sketch Generation

<p align="center">
  <img src="docs/swift_teaser.png" width="800px" alt="SwiftSketch Teaser"/>
</p>

*SwiftSketch is a diffusion model that generates vector sketches by denoising a Gaussian in stroke coordinate space. It generalizes effectively across diverse classes and takes under a second to produce a single high-quality sketch.*

#### Ellie Arar, Yarden Frenkel, Daniel Cohen-Or, Ariel Shamir, Yael Vinker

> Recent advancements in large vision-language models have enabled highly expressive and diverse vector sketch generation. However, state-of-the-art methods rely on a time-consuming optimization process involving repeated feedback from a pretrained model to determine stroke placement. Consequently, despite producing impressive sketches, these methods are limited in practical applications. In this work, we introduce SwiftSketch, a diffusion model for image-conditioned vector sketch generation that can produce high-quality sketches in less than a second. SwiftSketch operates by progressively denoising stroke control points sampled from a Gaussian distribution. Its transformer-decoder architecture is designed to effectively handle the discrete nature of vector representation and capture the inherent global dependencies between strokes. To train SwiftSketch, we construct a synthetic dataset of image-sketch pairs, addressing the limitations of existing sketch datasets, which are often created by non-artists and lack professional quality. For generating these synthetic sketches, we introduce ControlSketch, a method that enhances SDS-based techniques by incorporating precise spatial control through a depth-aware ControlNet. We demonstrate that SwiftSketch generalizes across diverse concepts, efficiently producing sketches that combine high fidelity with a natural and visually appealing style.

<p align="center">
  <a href="https://arxiv.org/abs/2502.08642"><img src="https://img.shields.io/badge/arXiv-2502.08642-b31b1b.svg" alt="arXiv"/></a>
  <a href="https://swiftsketch.github.io/"><img src="https://img.shields.io/static/v1?label=Project&message=Website&color=red" height="20.5" alt="Project Website"/></a>
</p>

[**Download the ControlSketch dataset**](https://drive.google.com/drive/folders/1L5kubR416QoTD_UAqH2FtSgNL4leUcys)

---

## 🔥 NEWS
- **`2025/09/27`**: The SwiftSketch code is released!
- **`2025/04/29`**: The ControlSketch code is released!
- **`2025/02/12`**: The ControlSketch dataset is released!
- **`2025/02/12`**: Paper is out!

---

## 📋 Table of Contents
- [Installation](#installation)
- [ControlSketch (Optimization)](#controlsketch)
- [Data Creation (SDXL)](#data-creation)
- [SwiftSketch (Diffusion Model)](#swiftsketch)
  - [Download Pretrained Models](#download-the-pretrained-models)
  - [SwiftSketch Generation](#swiftsketch-generation)
  - [SwiftSketch Training](#swiftsketch-training)
- [Citation](#citation)

---

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/swiftsketch/swiftsketch.git
   cd swiftsketch
   ```

2. **Create a new conda environment:**
   ```bash
   conda create -n swiftsketch_env python=3.9.19 -y
   conda activate swiftsketch_env
   ```

3. **Install diffvg:**  
   Please follow the [DiffVG installation guide](https://github.com/BachiLi/diffvg?tab=readme-ov-file#install).

4. **Install required dependencies:**
   ```bash
   pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121
   pip install -r requirements.txt
   pip install git+https://github.com/openai/CLIP.git
   ```

---

## ControlSketch

Navigate to the `ControlSketch` directory:
```bash
cd ControlSketch
```

To sketch your own image using the optimization-based method, **ControlSketch**, run:
```bash
python object_sketching.py --target <file_path>
```

The `--target` can be one of the following:
1. An image file (e.g., `.png`, `.jpg`).
2. A dictionary file (`.npy` or `.npz`) created by `make_sdxl_data.py`, containing keys: `image`, `mask`, `attn_map`, and `caption`.

The final sketch will be saved in the `output_sketches/` folder. If the input is a dictionary, the sketch will also be saved inside the dictionary.

### Optional Arguments:
* `--num_strokes`: Defines the number of strokes used to create the sketch, determining the level of abstraction (default: `32`). Different images may benefit from different stroke counts.
* `--fix_scale`: If your image is not square (1:1), set this flag to `1` to automatically fix the scale without resizing or distorting the image.
* `--object_name`: Target word for extracting the object cross-attention map for stroke initialization. If omitted, CLIP attention is used.
* `--caption`: A precise caption describing the object and its position. If omitted, a captioning model (BLIP-2) is used to generate one automatically.
* `--use_cpu`: Run optimization on CPU (not recommended, as it is significantly slower).

### Examples:

- **Sketching an image with default parameters:**
  ```bash
  python object_sketching.py --target "./data/lion.png"
  ```

- **Sketching with a specified object name and caption:**
  ```bash
  python object_sketching.py --target "./data/lion.png" --object_name "lion" --caption "lion standing"
  ```

- **Sketching using a dictionary target:**
  ```bash
  python object_sketching.py --target "./data/cat.npz"
  ```

- **Sketching a non-square image using `fix_scale`:**
  ```bash
  python object_sketching.py --target "./data/elephant.png" --fix_scale 1
  ```

---

## Data Creation

To generate a synthetic data sample using SDXL, run:
```bash
python make_sdxl_data.py --obj "<object to generate>"
```

Each generated sample is saved as a dictionary containing the following keys:
- **`image`**: The generated RGB image.
- **`mask`**: The corresponding segmentation mask.
- **`attn_map`**: The extracted cross-attention map.
- **`caption`**: The caption of the generated image.

Generated samples are saved in the `SDXL_samples/` directory by default.

### Optional Arguments:
* `--num_of_samples`: Number of samples to generate for the given object using different seeds (default: `1`).
* `--save_compressed_dict`: If set to `0`, output dictionary is saved in uncompressed `.npy` format. Default is `1` (compressed `.npz`).
* `--output_dir`: Directory to save the output dictionaries (default: `SDXL_samples/`).

### Example:
Generate 10 samples of a cat and save them to a specified directory:
```bash
python make_sdxl_data.py --obj "cat" --output_dir "path/to/output/dir" --num_of_samples 10
```

---

## SwiftSketch

Navigate to the `SwiftSketch` directory:
```bash
cd SwiftSketch
```

### Download the Pretrained Models

Download the pretrained models, unzip them, and place them under `./save/`:

- [**sketch-diffusion**](https://drive.google.com/uc?export=download&id=19FryO99dCmz-Dw1jzeZITUI0uuksiOA-)
- [**refinement-network**](https://drive.google.com/uc?export=download&id=1OrLzwaJXZ4SlDw3hqn71Yg1L01ytLv2x)

### SwiftSketch Generation

To generate sketches using SwiftSketch inference, run:
```bash
python -m generate \
  --model_path "<path/to/sketch-diffusion_model.pt>" \
  --refine_model_path "<path/to/refinement-network.pt>" \
  --input_data "<path/to/input>" \
  --output_dir "<path/to/output>"
```

The `--input_data` can be:
1. A single image file.
2. A directory of images.
3. An `.npy`/`.npz` dictionary containing the `'image'` key.
4. A directory of `.npy`/`.npz` dictionaries.

**Notes:**
- The final sketches are saved in the specified `--output_dir`. If omitted, output is saved in an `output_sketches/` folder inside the input directory.
- If the input is a dictionary, the sketch is also appended to the dictionary.

### Example:
Generate sketches for all images in the `examples/` directory:
```bash
python -m generate \
  --model_path "./save/sketch-diffusion/model000450000.pt" \
  --refine_model_path "./save/refinement-network/model000430000.pt" \
  --input_data "./examples" \
  --output_dir "./output_sketches"
```

Example outputs can be inspected in the `output_sketches/` folder.

---

### SwiftSketch Training

Training SwiftSketch is a 3-step pipeline:

#### 1. Extract Image Features
Prepare data for training by computing and caching image features into input dictionaries:
```bash
python -m utils.get_features --dir_name <path/to/data>
```
- `--dir_name`: Path to a directory containing `.npy`/`.npz` dictionaries.
- Adds the `'image_features'` key to each dictionary.

#### 2. Train Sketch Diffusion Model
Train the core diffusion model on vectorized sketch targets:
```bash
python -m train.train_SwiftSketch \
  --save_dir <path/to/save_dir> \
  --train_data_dir <path/to/training_data>
```
- Model checkpoints and cached data are saved to `--save_dir`.
- `--train_data_dir`: One or more paths to training data directories.

**Optional Arguments:**
* `--num_steps`: Total number of training steps.
* `--batch_size`: Batch size during training.
* `--save_interval`: Save checkpoints every $N$ steps.
* `--data_name`: Filename for the cached dataset.
* `--cat_data_size`: Maximum number of files to use per category/input directory.
* `--target_key_name`: Target SVG key in dictionaries (default: `"svg_32s"`, matching ControlSketch output).

**Example:**  
Train for 50,000 steps on 1,000 samples each of cat and dog categories:
```bash
python -m train.train_SwiftSketch \
  --save_dir "./save/cat_dog_model" \
  --num_steps 50000 \
  --data_name "cat_dog_data" \
  --cat_data_size 1000 \
  --batch_size 16 \
  --train_data_dir "./controlsketch_data/train/cat" "./controlsketch_data/train/dog"
```

#### 3. Train Refinement Network
First, generate intermediate diffusion sketches and save them into the input dictionaries:
```bash
python -m generate \
  --model_path "<path/to/sketch-diffusion-model.pt>" \
  --use_refine 0 \
  --save_diffusion_sketch_in_dict 1 \
  --input_data "<path/to/input>"
```
This adds the `'svg_diffusion'` key to each input dictionary.

Next, train the refinement network:
```bash
python -m refine_model.train_refine.train_refine_model \
  --save_dir "<path/to/save_dir>" \
  --resume_checkpoint "<path/to/pretrained_sketch_diffusion_model.pt>" \
  --train_data_dir "<path/to/training_data>"
```

- Checkpoints and cached data will be saved to `--save_dir`.
- `--resume_checkpoint`: Path to the pretrained Sketch Diffusion checkpoint (e.g., `model000450000.pt`).

**Optional Arguments:**
* `--num_steps`: Number of training steps.
* `--batch_size`: Batch size during training.
* `--save_interval`: Save checkpoints every $N$ steps.
* `--data_name`: Filename for cached data.
* `--cat_data_size`: Maximum number of samples per category.
* `--target_key_name`: Name of target SVG key (default: `"svg_32s"`).
* `--diffusion_key_name`: Name of intermediate diffusion SVG key (default: `"svg_diffusion"`).

**Example:**  
Train the refinement network for 10,000 steps initialized from the base diffusion checkpoint:
```bash
python -m refine_model.train_refine.train_refine_model \
  --save_dir "./save/cat_dog_refine_model" \
  --resume_checkpoint "./save/sketch-diffusion/model000450000.pt" \
  --num_steps 10000 \
  --data_name "cat_dog_data" \
  --cat_data_size 1000 \
  --batch_size 16 \
  --train_data_dir "./controlsketch_data/train/cat" "./controlsketch_data/train/dog"
```

---

## Citation

If you make use of our work, please cite our paper:

```bibtex
@inproceedings{10.1145/3721238.3730612,
  author    = {Arar, Ellie and Frenkel, Yarden and Cohen-Or, Daniel and Shamir, Ariel and Vinker, Yael},
  title     = {SwiftSketch: A Diffusion Model for Image-to-Vector Sketch Generation},
  year      = {2025},
  isbn      = {9798400715402},
  publisher = {Association for Computing Machinery},
  address   = {New York, NY, USA},
  url       = {https://doi.org/10.1145/3721238.3730612},
  doi       = {10.1145/3721238.3730612},
  booktitle = {Proceedings of the Special Interest Group on Computer Graphics and Interactive Techniques Conference Conference Papers},
  articleno = {82},
  numpages  = {12},
  keywords  = {Sketch Synthesis, Image-to-Vector Generation, Image-based Rendering, Vector Graphics, Diffusion Models, Stroke-based Representation},
  series    = {SIGGRAPH Conference Papers '25}
}
```
