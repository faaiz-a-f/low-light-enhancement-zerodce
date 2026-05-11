# Zero-DCE++ with Channel Attention (SE-Block)
## University Final Project - Low-Light Image Enhancement

### 📋 Project Overview

**Novelty:** Channel Attention (SE-Block) added after Conv3 in Zero-DCE++ encoder  
**Dataset:** LOL-v2-real (689 paired images)  
**Targets:** PSNR > 19 dB, SSIM > 0.65, LPIPS < 0.30  
**GPU:** RTX 3060 (~2.5 hours for 200 epochs)

### 🏗️ Architecture

```
Input (B, 3, 512, 512)
  ↓
Conv1(3→32) → Conv2(32→32) → Conv3(32→32)
  ↓
[SE-BLOCK] ← YOUR NOVELTY (Channel Attention)
  ↓
Conv4(32→32)
  ↓
[Decoder with Skip Connections]
  ↓
Conv7 → Output: 24 channels (8 curves × 3 RGB)
  ↓
Curve Application: LE(x; α) = x + α·x·(1-x), applied 8 times
  ↓
Enhanced Image (B, 3, 512, 512)
```

### 📦 Installation

```bash
# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

# Optional: Install LPIPS and NIQE for full evaluation
pip install lpips piq
```

### 🚀 Running the Project

#### Step 1: Training
```bash
python train_zerodce_seblock.py
```
- Trains for 200 epochs with CosineAnnealingLR
- Saves best model to `best_zerodce_seblock.pth`
- Generates `training_history.png`

**Training Parameters:**
- Optimizer: Adam
- Learning Rate: 1e-4
- Weight Decay: 1e-4
- Scheduler: CosineAnnealingLR, T_max=200
- Batch Size: 8
- Image Size: 512×512 (random crop for training)

#### Step 2: Evaluation
```bash
python evaluate_zerodce_seblock.py
```
- Evaluates on LOL-v2 test set
- Calculates PSNR, SSIM, LPIPS, NIQE
- Generates visualizations:
  - `evaluation_results.png` - Side-by-side comparisons
  - `metrics_distribution.png` - Histograms

### 📊 Loss Functions (Zero-Reference)

```
L_total = 1×L_spa + 10×L_exp + 5×L_col

L_spa:  Spatial consistency (directional gradient penalty)
L_exp:  Exposure control (target E=0.6)
L_col:  Color constancy (Gray World assumption)
```

Note: Zero-reference means NO ground truth needed during training (only uses low-light images)

### 🎯 Evaluation Metrics

| Metric | Lower/Higher | Target | Library |
|--------|-------------|--------|---------|
| PSNR   | Higher (dB) | > 19.00| scikit-image |
| SSIM   | Higher      | > 0.65 | scikit-image |
| LPIPS  | Lower       | < 0.30 | lpips |
| NIQE   | Lower       | < 3.8  | piq |

### 📁 Project Structure

```
low-light-enhancement-zerodce/
├── train_zerodce_seblock.py          # Training script
├── evaluate_zerodce_seblock.py       # Evaluation script
├── requirements.txt                  # Dependencies
├── LOL-v2/                           # Dataset
│   └── Real_captured/
│       ├── Train/
│       │   ├── Low/                  # 689 low-light images
│       │   └── Normal/               # 689 ground truth images
│       └── Test/
│           ├── Low/                  # 100 low-light test images
│           └── Normal/               # 100 ground truth test images
├── best_zerodce_seblock.pth          # Trained model (saved after training)
├── training_history.png              # Training curves (after training)
└── evaluation_results.png            # Visual results (after evaluation)
```

### 🔍 Key Implementation Details

**SE-Block (Novelty):**
- Squeeze: Global average pooling
- Excitation: 2-layer FC with ReLU and Sigmoid
- Reduction ratio: 8 (channels reduced by 8×)
- Applied after Conv3 in encoder

**Curve Application:**
- 8 sequential curve applications
- Each curve: α ∈ [-1, 1] (from Tanh output)
- Formula ensures pixel values stay in [0, 1]

**Data Loading:**
- Training: Random 512×512 crops for augmentation
- Testing: Center crop to 512×512
- Normalization: ToTensor() → [0, 1]

### 📈 Expected Results

**On LOL-v2 Test Set (100 images):**
- PSNR: ~18-20 dB
- SSIM: ~0.60-0.70
- LPIPS: ~0.25-0.35 (if available)

### ⚙️ Hyperparameter Tuning

To improve results:

1. **Increase training time:**
   ```python
   NUM_EPOCHS = 300  # Default: 200
   ```

2. **Adjust learning rate:**
   ```python
   LEARNING_RATE = 5e-5  # Default: 1e-4 (smaller = slower but more stable)
   ```

3. **Change batch size:**
   ```python
   BATCH_SIZE = 16  # Default: 8 (larger = faster training, needs more VRAM)
   ```

4. **Modify image size:**
   ```python
   IMG_SIZE = 256  # Default: 512 (smaller = faster, less detail)
   ```

### 🐛 Troubleshooting

**CUDA not detected:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**Out of memory (OOM):**
- Reduce batch size: `BATCH_SIZE = 4` or `2`
- Reduce image size: `IMG_SIZE = 256`

**LPIPS/NIQE import errors:**
```bash
pip install lpips piq
```

### 📝 Output Files

After training:
- `best_zerodce_seblock.pth` - Model weights
- `training_history.png` - Loss curves

After evaluation:
- `evaluation_results.png` - Visual comparisons (Low → Enhanced → Ground Truth)
- `metrics_distribution.png` - PSNR/SSIM histograms

### 🔗 References

- Zero-DCE (Original): Guo et al., CVPR 2020
- Zero-DCE++: Li et al., IEEE TPAMI 2022
- SE-Net (Channel Attention): Hu et al., CVPR 2018
- LOL Dataset: Wei et al., BMVC 2018

### 🎓 Academic Notes

This implementation follows the official Zero-DCE++ specification with your novelty contribution (SE-Block). The zero-reference training approach means:
- No paired images needed (only low-light images)
- Unsupervised learning using image statistics
- More practical for real-world scenarios
