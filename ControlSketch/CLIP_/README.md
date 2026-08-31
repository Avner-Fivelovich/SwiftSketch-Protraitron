# CLIP: Contrastive Language-Image Pre-Training

[![Blog](https://img.shields.io/badge/OpenAI-Blog-blue.svg)](https://openai.com/blog/clip/)
[![Paper](https://img.shields.io/badge/arXiv-2103.00020-b31b1b.svg)](https://arxiv.org/abs/2103.00020)
[![Model Card](https://img.shields.io/badge/Model%20Card-Markdown-green.svg)](model-card.md)
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/openai/clip/blob/master/notebooks/Interacting_with_CLIP.ipynb)

**CLIP** (Contrastive Language-Image Pre-Training) is a neural network trained on a wide variety of `(image, text)` pairs. It can be instructed in natural language to predict the most relevant text snippet given an image, without directly optimizing for the downstream task—similar to the zero-shot capabilities of GPT-2 and GPT-3.

CLIP matches the performance of the original ResNet-50 on ImageNet "zero-shot" without using any of the original 1.28M labeled training examples, overcoming several major challenges in computer vision robustness and generalizability.

> [!NOTE]
> This repository fork includes custom forward and backward attention hooks on the visual transformer (`ResidualAttentionBlock` and `multi_head_attention_forward`), enabling gradient-weighted token relevancy maps and attention interpretability visualizations.

---

## Table of Contents

- [Approach](#approach)
- [Installation](#installation)
  - [Prerequisites](#prerequisites)
  - [Install Dependencies](#install-dependencies)
  - [Install CLIP](#install-clip)
- [Quickstart Usage](#quickstart-usage)
- [API Reference](#api-reference)
  - [Module Functions](#module-functions)
  - [Model Methods](#model-methods)
- [Examples](#examples)
  - [Zero-Shot Classification](#zero-shot-classification)
  - [Linear-Probe Evaluation](#linear-probe-evaluation)
  - [Attention Interpretability & Visualization](#attention-interpretability--visualization)
- [Notebooks](#notebooks)
- [License & Citation](#license--citation)

---

## Approach

![CLIP Overview](https://raw.githubusercontent.com/openai/CLIP/main/CLIP.png)

CLIP jointly trains an image encoder and a text encoder to predict the correct pairings of a batch of `(image, text)` training examples. At test time, the learned visual and text representations enable zero-shot transfer of the model to downstream classification tasks via prompt engineering (e.g., `"a photo of a {label}"`).

---

## Installation

### Prerequisites

- Python 3.7+
- PyTorch >= 1.7.1
- TorchVision >= 0.8.2

### Install Dependencies

Install PyTorch and TorchVision following the official instructions at [pytorch.org](https://pytorch.org/get-started/locally/).

For CUDA GPU support:
```bash
conda install --yes -c pytorch pytorch torchvision cudatoolkit=11.0
pip install ftfy regex tqdm
```

*(Replace `cudatoolkit=11.0` with your appropriate CUDA runtime version or use `cpuonly` on machines without a dedicated GPU).*

For running the attention interpretability and visualization scripts ([`example.py`](example.py)), install OpenCV and Matplotlib:
```bash
pip install opencv-python matplotlib
```

### Install CLIP

To install directly from GitHub:
```bash
pip install git+https://github.com/openai/CLIP.git
```

Or install locally in editable mode from this directory:
```bash
pip install -e .
```

---

## Quickstart Usage

```python
import torch
import clip
from PIL import Image

# Select computation device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load model architecture and preprocessing transform
model, preprocess = clip.load("ViT-B/32", device=device)

# Prepare inputs
image = preprocess(Image.open("astronaut.png")).unsqueeze(0).to(device)
text = clip.tokenize(["a diagram", "a dog", "an astronaut"]).to(device)

with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)
    
    logits_per_image, logits_per_text = model(image, text)
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()

print("Label probabilities:", probs)
```

---

## API Reference

### Module Functions

The `clip` package exposes the following primary functions:

| Function | Signature | Description |
| :--- | :--- | :--- |
| `clip.available_models()` | `() -> List[str]` | Returns a list of available CLIP model architecture names (e.g., `['RN50', 'RN101', 'RN50x4', 'ViT-B/32']`). |
| `clip.load()` | `(name: str, device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu", jit: bool = True, download_root: str = None) -> Tuple[nn.Module, Callable]` | Downloads (if necessary) and loads the specified CLIP model alongside its TorchVision preprocessing pipeline. The `name` parameter can also be a path to a local checkpoint. |
| `clip.tokenize()` | `(text: Union[str, List[str]], context_length: int = 77, truncate: bool = False) -> torch.LongTensor` | Returns a `LongTensor` containing tokenized sequences for input string(s), padded or truncated to `context_length`. |

### Model Methods

The model instance returned by `clip.load()` provides:

| Method | Signature | Description |
| :--- | :--- | :--- |
| `encode_image` | `model.encode_image(image: Tensor) -> Tensor` | Encodes a batch of preprocessed images into normalized visual feature vectors. |
| `encode_text` | `model.encode_text(text: Tensor) -> Tensor` | Encodes a batch of tokenized text sequences into normalized language feature vectors. |
| `forward` | `model(image: Tensor, text: Tensor) -> Tuple[Tensor, Tensor]` | Computes cosine similarity logits `(logits_per_image, logits_per_text)` scaled by the learned temperature parameter (`* 100`). |

---

## Examples

### Zero-Shot Classification

Perform zero-shot prediction on the [CIFAR-100 dataset](https://www.cs.toronto.edu/~kriz/cifar.html) without task-specific fine-tuning:

```python
import os
import clip
import torch
from torchvision.datasets import CIFAR100

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Download CIFAR-100 test set
cifar100 = CIFAR100(root=os.path.expanduser("~/.cache"), download=True, train=False)

# Prepare image and text inputs
image, class_id = cifar100[3637]
image_input = preprocess(image).unsqueeze(0).to(device)
text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in cifar100.classes]).to(device)

# Compute embeddings
with torch.no_grad():
    image_features = model.encode_image(image_input)
    text_features = model.encode_text(text_inputs)

# Normalize and calculate cosine similarity
image_features /= image_features.norm(dim=-1, keepdim=True)
text_features /= text_features.norm(dim=-1, keepdim=True)
similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
values, indices = similarity[0].topk(5)

# Display top predictions
print("\nTop predictions:\n")
for value, index in zip(values, indices):
    print(f"{cifar100.classes[index]:>16s}: {100 * value.item():.2f}%")
```

**Expected output:**
```text
Top predictions:

           snake: 65.31%
          turtle: 12.29%
    sweet_pepper: 3.83%
          lizard: 1.88%
       crocodile: 1.75%
```

---

### Linear-Probe Evaluation

Extract frozen CLIP features and train a linear classifier using [scikit-learn](https://scikit-learn.org/):

```python
import os
import clip
import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR100
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Load CIFAR-100 train & test datasets
root = os.path.expanduser("~/.cache")
train = CIFAR100(root, download=True, train=True, transform=preprocess)
test = CIFAR100(root, download=True, train=False, transform=preprocess)

def get_features(dataset):
    all_features = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(DataLoader(dataset, batch_size=100)):
            features = model.encode_image(images.to(device))
            all_features.append(features)
            all_labels.append(labels)

    return torch.cat(all_features).cpu().numpy(), torch.cat(all_labels).cpu().numpy()

# Extract features
train_features, train_labels = get_features(train)
test_features, test_labels = get_features(test)

# Train Logistic Regression classifier
classifier = LogisticRegression(random_state=0, C=0.316, max_iter=1000, verbose=1)
classifier.fit(train_features, train_labels)

# Evaluate accuracy
predictions = classifier.predict(test_features)
accuracy = np.mean((test_labels == predictions).astype(float)) * 100.0
print(f"Accuracy = {accuracy:.3f}%")
```

> [!TIP]
> The regularization parameter `C` should typically be tuned on a validation split via hyperparameter search.

---

### Attention Interpretability & Visualization

This repository fork incorporates forward and backward attention hooks on the visual transformer self-attention blocks (`attn_probs` and `attn_grad`) to compute token-level relevance maps and visual explanations (see [`example.py`](example.py)):

```python
import torch
import clip
from PIL import Image
from example import interpret

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load ViT-B/32 with jit=False to enable gradient tracking on attention maps
model, preprocess = clip.load("ViT-B/32", device=device, jit=False)

image = preprocess(Image.open("astronaut.png")).unsqueeze(0).to(device)
text = clip.tokenize(["an astronaut", "a spaceship"]).to(device)

# Compute and display gradient-weighted attention relevance heatmaps
interpret(model=model, image=image, text=text, device=device, index=0)
```

To run the full demonstration suite with multi-object interpretation examples:
```bash
python example.py
```

---

## Notebooks

Interactive Jupyter notebooks are available under the [`notebooks/`](notebooks) directory:

- [`Interacting_with_CLIP.ipynb`](notebooks/Interacting_with_CLIP.ipynb): Interactive zero-shot classification, similarity matrix visualization, and image-text retrieval.
- [`Prompt_Engineering_for_ImageNet.ipynb`](notebooks/Prompt_Engineering_for_ImageNet.ipynb): Prompt engineering templates and ensemble evaluation for ImageNet.

---

## License & Citation

The original CLIP codebase and pre-trained weights are provided by OpenAI under the [MIT License](LICENSE).

If you use CLIP or this repository in your research, please cite the original paper:

```bibtex
@inproceedings{radford2021learning,
  title={Learning Transferable Visual Models From Natural Language Supervision},
  author={Radford, Alec and Kim, Jong Wook and Hallacy, Chris and Ramesh, Aditya and Goh, Gabriel and Agarwal, Sandhini and Sastry, Girish and Askell, Amanda and Mishkin, Pamela and Clark, Jack and Krueger, Gretchen and Sutskever, Ilya},
  booktitle={International Conference on Machine Learning},
  pages={8748--8763},
  year={2021},
  organization={PMLR}
}
```
