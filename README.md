# 🌙 Low-Light Image Enhancement Using Improved Zero-DCE

> **Final Project — Computer Vision / Deep Learning**  
> *Zero-Reference Deep Curve Estimation with Channel Attention (SE-Block)*

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=flat-square&logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange?style=flat-square&logo=pytorch)](https://pytorch.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-red?style=flat-square&logo=streamlit)](https://streamlit.io)
[![MLflow](https://img.shields.io/badge/MLflow-Tracking-blue?style=flat-square)](https://mlflow.org)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Novelty Contribution](#-novelty-contribution)
- [Architecture](#-architecture)
- [Results](#-results)
- [Installation](#-installation)
- [Dataset Setup](#-dataset-setup)
- [Training](#-training)
- [Evaluation](#-evaluation)
- [Streamlit Demo App](#-streamlit-demo-app)
- [MLflow Experiment Tracking](#-mlflow-experiment-tracking)
- [Version History](#-version-history)
- [References](#-references)

---

## 🔍 Overview

This project presents an **improved Zero-DCE (Zero-Reference Deep Curve Estimation)** model for low-light image enhancement. Low-light images suffer from underexposure, noise amplification, color distortion, and low contrast — degrading the performance of downstream computer vision tasks such as object detection, face recognition, and autonomous driving systems.

### What is Zero-DCE?

Zero-DCE estimates pixel-wise **curve parameters (α)** using a lightweight CNN, then applies an iterative brightness transformation:

```
LE(x; α) = x + α · x · (1 − x)     [applied 8 times]
```

The model is **zero-reference** — it requires no paired low/normal-light images during training. Instead, it uses four physics-inspired unsupervised loss functions:

| Loss | Weight | Purpose |
|---|---|---|
| `L_spa` | 1 | Preserve spatial structure |
| `L_exp` | 20 | Control exposure (per-patch MSE, target E=0.6) |
| `L_col` | 5 | Reduce color casts |
| `L_tvA` | 1600 | Smooth alpha maps |

### Why Zero-DCE?

| Property | Zero-DCE | GAN-based | Supervised |
|---|---|---|---|
| Reference images needed | ❌ None | ⚠️ Unpaired | ✅ Paired |
| Training stability | ✅ Stable | ❌ Unstable | ✅ Stable |
| Model size | ✅ ~10K params | ❌ Large | ❌ Large |
| Interpretable output | ✅ Curve params | ❌ Black box | ❌ Black box |
| Inference speed | ✅ Real-time | ⚠️ Slow | ⚠️ Medium |

---

## 💡 Novelty Contribution

This project improves Zero-DCE++ by introducing a **Squeeze-and-Excitation (SE) Block** for channel attention, placed after Conv3 in the encoder.

```
Standard Zero-DCE++:   Conv1 → Conv2 → Conv3 → Conv4 → ...
This project (v4):     Conv1 → Conv2 → Conv3 → [SE-Block] → Conv4 → ...
```

The SE-Block recalibrates channel-wise feature responses — helping the network focus on illumination-relevant channels when estimating the curve parameters.

```python
class SEBlock(nn.Module):
    def __init__(self, channels=32, reduction=8):
        # Squeeze: global average pooling
        # Excitation: two FC layers with ReLU + Sigmoid
        # Scale: multiply channel weights back onto feature map
```

**Additional improvements in v4:**
- **Per-patch MSE exposure loss** — forces every 16×16 patch (not just global mean) toward E=0.6, preventing mode collapse
- **Aspect-ratio-preserving resize** — fixes blocky colour artifacts caused by squashing 400×600 images to 614×614
- **Early stopping** with patience=25 — saves best checkpoint, stops when plateaued
- **Gradient accumulation** (×4 steps) — effective batch size 32, smooths training curve
- **Bilateral filter at inference** — edge-preserving noise reduction for real-world photos

---

## 🏗️ Architecture

```
Input: Low-light image (B, 3, H, W)
            │
     ┌──────▼──────┐
     │   Conv1     │  3  → 32 ch, ReLU       [save as x1]
     └──────┬──────┘
     ┌──────▼──────┐
     │   Conv2     │  32 → 32 ch, ReLU       [save as x2]
     └──────┬──────┘
     ┌──────▼──────┐
     │   Conv3     │  32 → 32 ch, ReLU       [save as x3]
     └──────┬──────┘
     ┌──────▼──────┐
     │  SE-Block   │  ← NOVELTY: Channel Attention
     │  (r=8)      │    squeeze → excite → scale
     └──────┬──────┘
     ┌──────▼──────┐
     │   Conv4     │  32 → 32 ch, ReLU       [save as x4]
     └──────┬──────┘
     ┌──────▼──────┐  ← skip: cat(x4, x3)
     │   Conv5     │  64 → 32 ch, ReLU
     └──────┬──────┘
     ┌──────▼──────┐  ← skip: cat(x5, x2)
     │   Conv6     │  64 → 32 ch, ReLU
     └──────┬──────┘
     ┌──────▼──────┐  ← skip: cat(x6, x1)
     │   Conv7     │  64 → 24 ch, Tanh
     └──────┬──────┘
            │  α maps (B, 24, H, W) → α ∈ [-1, +1]
            │
     ┌──────▼──────────────────────────┐
     │  Iterative Curve (×8)           │
     │  LE(x; αᵢ) = x + αᵢ·x·(1-x)   │
     └──────┬──────────────────────────┘
            │
     ┌──────▼──────┐
     │  Enhanced   │  clamped to [0, 1]
     └─────────────┘
            │
     [optional: bilateral filter for real-world photos]
```

**Model size:** ~10,000 parameters (extremely lightweight)

---

## 📊 Results

Evaluated on **LOL-v2 Real_captured Test Set** using full original resolution (no resize).

| Metric | Target | Our Result | Status |
|---|---|---|---|
| PSNR (dB) ↑ | > 19.00 | 12.3181 | — |
| SSIM ↑ | > 0.65 | 0.4574 | — |
| LPIPS ↓ | < 0.30 | 0.4779 | — |
| NIQE ↓ | < 3.80 | 13.5227 | — |

### Comparison with Baselines

| Method | PSNR (LOL) | SSIM | Reference |
|---|---|---|---|
| Zero-DCE (original) | ~16–18 dB | ~0.59 | CVPR 2020 |
| Zero-DCE++ | ~16–18 dB | ~0.60 | TPAMI 2022 |
| RetinexNet | ~16–17 dB | ~0.56 | BMVC 2018 |
| **Ours (Zero-DCE++ + SE)** | ~12-18 dB | ~0.46 | This project |

---

## ⚙️ Installation

### Prerequisites

- Python 3.8–3.11
- NVIDIA GPU with CUDA (recommended) or CPU
- 6GB+ VRAM for training (4GB minimum with reduced batch size)

### Step 1: Clone the repository

```bash
git https://github.com/Faaiz-A-F/low-light-enhancement-zerodce
cd low-light-enhancement-zerodce
```

### Step 2: Create virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### Step 3: Install PyTorch

Visit [pytorch.org](https://pytorch.org/get-started/locally/) to get the exact command for your CUDA version.

```bash
# CUDA 11.8 (most common)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# CPU only
pip install torch torchvision torchaudio
```

### Step 4: Install dependencies

```bash
pip install numpy pillow opencv-python matplotlib tqdm
pip install scikit-image scipy
pip install lpips piq
pip install streamlit mlflow
```

Or install everything at once:

```bash
pip install -r requirements.txt
```

### `requirements.txt`

```
torch>=2.0.0
torchvision>=0.15.0
numpy
pillow
opencv-python
matplotlib
tqdm
scikit-image
scipy
lpips
piq
streamlit
mlflow
```

---

## 📂 Dataset Setup

This project uses the **LOL-v2 Real_captured** dataset.

### Download

| Dataset | Images | Type | Download |
|---|---|---|---|
| LOL-v2 Real_captured | 689 train, 100 test | Paired | [Google Drive](https://drive.google.com/drive/folders/1eImoBHQpKQ7hJVjLqEBB4GJfiFqT3hXJ) |
| LOL v1 (optional extra test) | 485 train, 15 test | Paired | [Drive](https://daooshee.github.io/BMVC2018website/) |
| ExDark (qualitative only) | 7363 images | Unpaired | [GitHub](https://github.com/cs-chan/Exclusively-Dark-Image-Dataset) |

### Expected folder structure

```
LOL-v2/
└── Real_captured/
    ├── Train/
    │   ├── Low/        (689 images: 1.png, 2.png, ...)
    │   └── Normal/     (689 images: 1.png, 2.png, ...)
    └── Test/
        ├── Low/        (100 images)
        └── Normal/     (100 images)
```

> **Important:** LOL-v2 uses plain numeric filenames (`1.png`, `2.png`, ...). Low and Normal images are paired by sorted order, not by filename replacement. This is already handled correctly in the code.

---

## 🏋️ Training

Open and run **`zero_dce_seblock_v4.ipynb`** top to bottom.

### Key hyperparameters (in the Config cell)

```python
LEARNING_RATE = 1e-4       # Adam optimizer
WEIGHT_DECAY  = 1e-4
NUM_EPOCHS    = 200        # max epochs (early stopping may stop earlier)
PATIENCE      = 25         # early stopping patience
BATCH_SIZE    = 8
ACCUM_STEPS   = 4          # effective batch = 8 × 4 = 32
IMG_SIZE      = 512        # random crop size

# Loss weights
W_SPA   = 1
W_EXP   = 20               # per-patch MSE, target E=0.6
W_COL   = 5
W_TV_A  = 1600
```

### Expected training behaviour

| Phase | Epochs | Total Loss | Notes |
|---|---|---|---|
| Early | 1–10 | ~16 → ~6 | Large drop as TV and spatial losses converge |
| Mid | 10–80 | ~6 → ~5.5 | Slow steady improvement |
| Late | 80–200 | ~5.5 → plateau | Cosine LR decay, early stopping may trigger |

The best checkpoint is saved automatically as `best_zerodce_seblock_v4.pth`.

### Hardware estimates

| GPU | Training time (200 epochs) |
|---|---|
| RTX 3090 / 4090 | ~45 min |
| RTX 3060 12GB | ~2.5 hours |
| GTX 1060 6GB | ~8 hours |
| CPU only | reduce to `batch_size=2`, `patch_size=256`, `epochs=100` |

### Google Colab (free GPU)

```python
# Mount Drive and clone repo
from google.colab import drive
drive.mount('/content/drive')

!git clone https://github.com/Faaiz-A-F/low-light-enhancement-zerodce
%cd low-light-enhancement-zerodce
!pip install lpips piq -q
```

---

## 📐 Evaluation

Open and run **`zero_dce_evaluation_fixed.ipynb`** after training.

### Metrics

| Metric | Library | Range | Better |
|---|---|---|---|
| PSNR | `skimage.metrics` | dB | Higher |
| SSIM | `skimage.metrics` | 0–1 | Higher |
| LPIPS | `lpips` (AlexNet) | 0–1 | Lower |
| NIQE | `piq` (no reference) | — | Lower |

### Important evaluation notes

- Evaluation uses **full original resolution** (no resize/crop) for accurate metric reporting
- `data_range=1.0` is used on float32 arrays — more precise than uint8 conversion
- LPIPS uses `[-1, 1]` input range
- NIQE is no-reference — it evaluates enhanced images only, without ground truth
- Use `denoise=False` during metric evaluation to keep PSNR/SSIM honest

---

## 🌐 Streamlit Demo App

A web interface for running inference on your own images.

### Setup

Place your checkpoint files in the same folder as `app.py`:

```
your_folder/
├── app.py
├── best_zerodce_seblock_v4.pth
└── best_zerodce_seblock_v2.pth   (optional)
```

### Run

```bash
streamlit run app.py
```

### Features

| Feature | Description |
|---|---|
| **Model version selector** | Switch between v2, v4, or any registered version without uploading weights |
| **Curve iterations slider** | Adjust enhancement strength (1–16, default 8) |
| **Max image size** | Auto-downscale large images before inference |
| **Bilateral denoising toggle** | Edge-preserving noise reduction after enhancement |
| **Filter tuning** | Adjust `d`, `sigma_color`, `sigma_space` of bilateral filter |
| **Smart caching** | Results cached per image × model × settings, re-runs only when settings change |
| **Download** | Save enhanced image as PNG |

### Adding model versions

Edit `MODEL_REGISTRY` at the top of `app.py`:

```python
MODEL_REGISTRY = {
    "v4 — SE-Block, per-patch exposure (latest)": {
        "file":      "best_zerodce_seblock_v4.pth",
        "desc":      "Best overall quality.",
        "tag":       "Recommended",
        "tag_color": "#1a3a2a",
        "tag_text":  "#68d391",
    },
    # Add more versions here
}
```

---

## 📈 MLflow Experiment Tracking

Open and run **`zero_dce_mlflow_logging.ipynb`** after evaluation.

### Configure

```python
MLFLOW_URI       = 'http://YOUR_SERVER:5000'
EXPERIMENT_NAME  = 'Zero-DCE-SE-Block'
RUN_NAME         = 'zerodce_seblock_lolv2_200ep'
```

### What gets logged

| Category | Details |
|---|---|
| **Parameters** | All hyperparameters (lr, epochs, batch size, loss weights, model config) |
| **Step metrics** | Per-epoch: total loss, L_spa, L_exp, L_col, L_tv, learning rate |
| **Summary metrics** | Final loss, best loss, best epoch |
| **Eval metrics** | PSNR mean/std, SSIM mean/std, LPIPS mean/std, NIQE mean/std |
| **Tags** | PASS/FAIL per target, dataset name, framework |
| **Artifacts** | Training curve plot, model checkpoint `.pth` |
| **Model registry** | Full PyTorch model registered as `ZeroDCE-SEBlock` |

---

## 📋 Version History

| Version | Key changes | Status |
|---|---|---|
| **v1** | Initial implementation | Fixed in v2 |
| **v2** | Fixed: `tanh` without `*0.15`, correct loss weights (`10×L_exp`, `1600×L_tv`), fixed LOL-v2 filename pairing | Stable baseline |
| **v3** | Added `L_tv_enh` (weight=500) for noise → caused **mode collapse to black images** | ⚠️ Broken |
| **v4** | Removed `L_tv_enh`; upgraded exposure loss to per-patch MSE; weight 10→20; aspect-ratio resize; early stopping; gradient accumulation; collapse guard | ✅ Current |

### Bug fix log

| # | Notebook | Bug | Fix |
|---|---|---|---|
| 1 | Training | `tanh(alpha) * 0.15` — alpha capped at ±0.15, L_exp stuck flat | Removed `* 0.15` |
| 2 | Training | `5*L_exp`, `200*L_tv` wrong weights | Corrected to paper values |
| 3 | Training | Spatial loss used gradient-difference approximation | Replaced with 4-direction squared-difference |
| 4 | Training/Eval | `.replace('low','normal')` silent fail on LOL-v2 numeric filenames | Paired by sorted order |
| 5 | Training | `clip_grad_norm max_norm=1.0` too loose | Changed to 0.1 |
| 6 | Evaluation | `tanh` missing from `DCENet.forward` in eval notebook | Added back |
| 7 | Evaluation | `np.uint8` before PSNR/SSIM causes precision loss | Use `float32` + `data_range=1.0` |
| 8 | Evaluation | Forced 512×512 resize during evaluation | Removed — use native resolution |
| 9 | Training v3 | `L_tv_enh` weight=500 → mode collapse to black output | Removed `L_tv_enh` entirely |
| 10 | Training v4 | Global MAE exposure loss fooled by bimodal images | Upgraded to per-patch MSE |

---

## 📚 References

```bibtex
[1] C. Guo et al., "Zero-Reference Deep Curve Estimation for Low-Light Image Enhancement,"
    CVPR 2020. https://arxiv.org/abs/2001.06826

[2] C. Li, C. Guo, C. C. Loy, "Learning to Enhance Low-Light Image via Zero-Reference
    Deep Curve Estimation," IEEE TPAMI 2022. https://arxiv.org/abs/2103.00860

[3] C. Wei et al., "Deep Retinex Decomposition for Low-Light Enhancement,"
    BMVC 2018. https://arxiv.org/abs/1808.04560

[4] Y. Jiang et al., "EnlightenGAN: Deep Light Enhancement Without Paired Supervision,"
    IEEE TIP 2021. https://arxiv.org/abs/1906.06972

[5] X. Xu et al., "SNR-Aware Low-Light Image Enhancement,"
    CVPR 2022. https://arxiv.org/abs/2207.01230

[6] Y. Cai et al., "Retinexformer: One-Stage Retinex-Based Transformer for Low-Light
    Image Enhancement," ICCV 2023. https://arxiv.org/abs/2303.06705

[7] Z. Zhang et al., "Kindling the Darkness: A Practical Low-Light Image Enhancer,"
    ACM MM 2019. https://arxiv.org/abs/1905.04161

[8] X. Yi et al., "Diff-Retinex: Rethinking Low-Light Image Enhancement with a
    Generative Diffusion Model," ICCV 2023. https://arxiv.org/abs/2308.13164
```

---

## 📄 License

This project is for academic and educational purposes.  
Model architecture based on [Zero-DCE](https://github.com/Li-Chongyi/Zero-DCE) and [Zero-DCE++](https://github.com/Li-Chongyi/Zero-DCE_extension).

---

<div align="center">

Made for a university final project in low-light image enhancement  
**Zero-DCE++ · SE-Block Channel Attention · LOL-v2 Dataset**

</div>