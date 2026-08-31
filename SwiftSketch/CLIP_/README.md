# Contrastive Language-Image Pre-Training (CLIP)

[![OpenAI Blog](https://img.shields.io/badge/OpenAI-Blog-412991.svg)](https://openai.com/blog/clip/)
[![Paper](https://img.shields.io/badge/arXiv-2103.00020-B31B1B.svg)](https://arxiv.org/abs/2103.00020)
[![Model Card](https://img.shields.io/badge/Model-Card-007ACC.svg)](model-card.md)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/openai/clip/blob/master/notebooks/Interacting_with_CLIP.ipynb)

CLIP (Contrastive Language-Image Pre-Training) is a neural network trained on a wide variety of `(image, text)` pairs. It can be instructed in natural language to predict the most relevant text snippet given an image without directly optimizing for the task—similar to the zero-shot capabilities of GPT-2 and GPT-3. 

CLIP matches the performance of the original ResNet-50 on ImageNet zero-shot without using any of the original 1.28M labeled examples, overcoming several major challenges in computer vision such as task-specific dataset bias and poor out-of-distribution generalization.

---

## Table of Contents

- [Approach](#approach)
- [Installation](#installation)
- [Usage](#usage)
- [API Reference](#api-reference)
  - [Top-level Functions](#top-level-functions)
  - [Model Methods](#model-methods)
- [Available Models](#available-models)
- [Examples](#examples)
  - [Zero-Shot Prediction](#zero-shot-prediction)
  - [Linear-Probe Evaluation](#linear-probe-evaluation)
  - [Attention & Relevance Interpretation](#attention--relevance-interpretation)
- [Citation](#citation)

---

## Approach

![CLIP Approach Diagram](https://raw.githubusercontent.com/openai/CLIP/main/CLIP.png)

CLIP trains an image encoder and a text encoder simultaneously to predict which images were paired with which texts across a massive multimodal dataset.

---

## Installation

### Prerequisites
- Python >= 3.7
- PyTorch >= 1.7.1
- Torchvision >= 0.8.2

### Conda Environment Setup

```bash
# Create and configure a conda environment
conda install --yes -c pytorch pytorch=1.7.1 torchvision cudatoolkit=11.0
pip install ftfy regex tqdm

# Install the local package in editable mode
pip install -e .
```

> [!NOTE]
> Replace `cudatoolkit=11.0` with the appropriate CUDA version for your machine, or use `cpuonly` when installing on a CPU-only system.

---

## Usage

Below is a basic example demonstrating how to encode an image and text snippets, and compute class probabilities:

```python
import torch
import clip
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Load and preprocess image and text
image = preprocess(Image.open("astronaut.png")).unsqueeze(0).to(device)
text = clip.tokenize(["a diagram", "a dog", "an astronaut"]).to(device)

with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)
    
    logits_per_image, logits_per_text = model(image, text)
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()

print("Label probs:", probs)
```

---

## API Reference

### Top-level Functions

#### `clip.available_models() -> List[str]`
Returns the names of available CLIP model architectures.

#### `clip.load(name: str, device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu", jit: bool = True) -> Tuple[torch.nn.Module, Callable]`
Loads and returns:
1. The PyTorch CLIP model corresponding to `name` (or a path to a local checkpoint).
2. The TorchVision transformation pipeline required by the model.

If `jit=False`, loads the non-JIT (pure PyTorch) version of the model.

#### `clip.tokenize(text: Union[str, List[str]], context_length: int = 77) -> torch.LongTensor`
Returns a `LongTensor` containing tokenized sequences of the input text(s) padded/truncated to `context_length`.

---

### Model Methods

The model instance returned by `clip.load()` provides:

#### `model.encode_image(image: torch.Tensor) -> torch.Tensor`
Given a batch of preprocessed images of shape `[N, C, H, W]`, returns the normalized feature representations encoded by the vision backbone.

#### `model.encode_text(text: torch.Tensor) -> torch.Tensor`
Given a batch of tokenized text sequences of shape `[N, context_length]`, returns the normalized feature representations encoded by the language backbone.

#### `model.forward(image: torch.Tensor, text: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]`
Returns `(logits_per_image, logits_per_text)`. These logits represent the scaled cosine similarities between image and text features (scaled by `100.0`).

---

## Available Models

| Model Name | Backbone Architecture | Description |
| :--- | :--- | :--- |
| `RN50` | ResNet-50 | Standard ResNet-50 visual encoder |
| `RN101` | ResNet-101 | Deeper ResNet-101 visual encoder |
| `RN50x4` | ResNet-50 (4x scaling) | Scaled ResNet with 4x compute (EfficientNet-style) |
| `ViT-B/32` | Vision Transformer (ViT-B/32) | Base Vision Transformer with patch size 32x32 |

---

## Examples

### Zero-Shot Prediction

This example evaluates zero-shot classification on the [CIFAR-100 dataset](https://www.cs.toronto.edu/~kriz/cifar.html) as detailed in the paper:

```python
import os
import clip
import torch
from torchvision.datasets import CIFAR100

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Download and load dataset
cifar100 = CIFAR100(root=os.path.expanduser("~/.cache"), download=True, train=False)

# Prepare single sample and full class prompt list
image, class_id = cifar100[3637]
image_input = preprocess(image).unsqueeze(0).to(device)
text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in cifar100.classes]).to(device)

with torch.no_grad():
    image_features = model.encode_image(image_input)
    text_features = model.encode_text(text_inputs)

# Normalize features and compute cosine similarity
image_features /= image_features.norm(dim=-1, keepdim=True)
text_features /= text_features.norm(dim=-1, keepdim=True)
similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
values, indices = similarity[0].topk(5)

print("\nTop predictions:\n")
for value, index in zip(values, indices):
    print(f"{cifar100.classes[index]:>16s}: {100 * value.item():.2f}%")
```

Sample output:
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

You can train a fast linear classifier on top of frozen CLIP features using [scikit-learn](https://scikit-learn.org/):

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

train_features, train_labels = get_features(train)
test_features, test_labels = get_features(test)

classifier = LogisticRegression(random_state=0, C=0.316, max_iter=1000, verbose=1)
classifier.fit(train_features, train_labels)

predictions = classifier.predict(test_features)
accuracy = np.mean((test_labels == predictions).astype(float)) * 100.0
print(f"Linear Probe Accuracy: {accuracy:.2f}%")
```

---

### Attention & Relevance Interpretation

This repository includes support for computing and visualizing attention rollouts and relevancy maps (Class Activation Maps) for ViT backbones via `example.py`.

To run the visual interpretation script:

```bash
python example.py
```

---

## Citation

If you use CLIP in your research, please cite the original paper:

```bibtex
@inproceedings{Radford2021LearningTV,
  title={Learning Transferable Visual Models From Natural Language Supervision},
  author={Alec Radford and Jong Wook Kim and Chris Hallacy and Aditya Ramesh and Gabriel Goh and Sandhini Agarwal and Girish Sastry and Amanda Askell and Pamela Mishkin and Jack Clark and Gretchen Krueger and Ilya Sutskever},
  booktitle={ICML},
  year={2021}
}
```
