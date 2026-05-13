# CIFAR-100 Robustness Challenge

A three-part deep learning project for BU's CS541, exploring how different modeling approaches handle distribution shift on CIFAR-100. Each part progressively leverages more powerful priors: scratch-built CNNs → pretrained CNNs → pretrained Vision Transformers.

---

## 🏆 Leaderboard Results

| Part | Model | Clean Accuracy | OOD Accuracy |
|------|-------|---------------|--------------|
| 1 | Custom ResNet-style CNN | 76.66% | 64.9% |
| 2 | Fine-tuned ResNet50  | 83.3% | 75% |
| 3 | Fine-tuned transformer - dinov2 | 91% | 80.3% |

---

## Part 1: CNN from Scratch

**Goal:** Build and train a CNN entirely from scratch on CIFAR-100 (no pretrained weights).

### Architecture

A ResNet-inspired CNN with 8 residual blocks and stochastic depth:

```
Input (3×32×32)
→ Conv(3→64) + BN + ReLU
→ ResBlock(64→64)
→ ResBlock(64→128, stride=2)
→ ResBlock(128→128)
→ ResBlock(128→256, stride=2)
→ ResBlock(256→256)
→ ResBlock(256→512, stride=2)
→ ResBlock(512→512)
→ ResBlock(512→512)
→ AdaptiveAvgPool → Linear(512→256) → BN → ReLU → Dropout(0.5)
→ Linear(256→100)
```

Each `ResBlock` includes:
- Two 3×3 convolutions with BatchNorm
- Stochastic depth (drop path) for regularization
- 1×1 projection shortcut when dimensions change

### Training Strategy

| Hyperparameter | Value |
|---------------|-------|
| Epochs | 150 |
| Batch size | 128 |
| Optimizer | SGD + Nesterov momentum (0.9) |
| Learning rate | 0.1 → cosine decay → 1e-6 |
| LR warmup | Linear over 15 epochs |
| Weight decay | 5e-4 (excluding BN/bias layers) |
| Loss | CrossEntropy + label smoothing (0.1) |

### Data Augmentation

```python
transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.RandomGrayscale(p=0.1),
    transforms.AugMix(),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
    transforms.RandomApply([transforms.GaussianBlur(3)], p=0.4),
    transforms.RandomErasing(p=0.25),
])
```

### Inference

Test-Time Augmentation (TTA) with 6 augmented passes averaged at inference, yielding a ~1% accuracy boost over single-pass evaluation.

## Part 1: CNN from Scratch
 
**Goal:** Build and train a CNN entirely from scratch on CIFAR-100 (no pretrained weights).
 
### Architecture
 
A ResNet-inspired CNN with 8 residual blocks and stochastic depth:
 
```
Input (3×32×32)
→ Conv(3→64) + BN + ReLU
→ ResBlock(64→64)
→ ResBlock(64→128, stride=2)
→ ResBlock(128→128)
→ ResBlock(128→256, stride=2)
→ ResBlock(256→256)
→ ResBlock(256→512, stride=2)
→ ResBlock(512→512)
→ ResBlock(512→512)
→ AdaptiveAvgPool → Linear(512→256) → BN → ReLU → Dropout(0.5)
→ Linear(256→100)
```
 
Each `ResBlock` includes two 3×3 convolutions with BatchNorm, stochastic depth (drop path) for regularization, and a 1×1 projection shortcut when dimensions change.
 
### Training Strategy
 
| Hyperparameter | Value |
|---------------|-------|
| Epochs | 150 |
| Batch size | 128 |
| Optimizer | SGD + Nesterov momentum (0.9) |
| Learning rate | 0.1 → cosine decay → 1e-6 |
| LR warmup | Linear over 15 epochs |
| Weight decay | 5e-4 (excluding BN/bias layers) |
| Loss | CrossEntropy + label smoothing (0.1) |
 
### Data Augmentation
 
```python
transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.RandomGrayscale(p=0.1),
    transforms.AugMix(),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
    transforms.RandomApply([transforms.GaussianBlur(3)], p=0.4),
    transforms.RandomErasing(p=0.25),
])
```
 
### Experiment Progression
 
| Configuration | Train Acc | Val Acc | Clean Test | OOD Test |
|--------------|-----------|---------|------------|----------|
| Residual blocks + cosine LR, 30 epochs | 90% | 60% | — | 38% |
| + SGD + dropout + more augmentations, 100 epochs | 88% | 66% | 71% | 43.8% |
| + RandAugment, 100 epochs | 87% | 68% | 75% | — |
| + Label smoothing + repositioned residual blocks, 120 epochs | 91% | 69% | 76.8% | 55.6% |
| **Final: drop path + ResNet-style blocks + no BN weight decay, 150 epochs** | **89%** | **71%** | **76.8%** | **64.9%** |
 
### Key Findings
 
The most impactful improvements came from increasing model depth with additional residual blocks, applying stronger regularization (dropout, weight decay, label smoothing), using aggressive data augmentation (AugMix, RandAugment), and the warmup + cosine annealing LR schedule. OOD accuracy improved from 38% to 64.9% across all experiments — a 27% gain — showing that generalization to distorted images was the primary driver of design decisions.
 
---
 
## Part 2: Fine-Tuned CNN (ResNet50)
 
**Goal:** Fine-tune a large-scale pretrained CNN on CIFAR-100.
 
### Model Adaptations
 
ResNet50 pretrained on ImageNet was adapted for CIFAR-100's 32×32 images with two key modifications:
- Replaced the first 7×7 conv with a 3×3 conv and removed the max pooling layer, to prevent excessive early downsampling on small images
- Replaced the classification head with a two-layer dropout classifier (dropout rates 0.5 and 0.3) with a 512-dimensional hidden layer
### Training Strategy
 
| Hyperparameter | Value |
|---------------|-------|
| Epochs | 100 |
| Optimizer | SGD + Nesterov momentum |
| Weight decay | 5e-4 (non-BN parameters only) |
| Label smoothing | 0.2 |
| LR schedule | 5-epoch linear warmup + cosine annealing |
 
AugMix was used as the primary augmentation strategy alongside random flips, random crops, color jitter, random grayscale, Gaussian blur, and random erasing.
 
### Results
 
| Metric | Value |
|--------|-------|
| Train Accuracy | 80% |
| Validation Accuracy | 69% |
| Clean Test Accuracy | 83.3% |
| OOD (Hidden) Test Accuracy | 75% |
 
### Key Findings
 
Fine-tuned ResNet50 showed strong clean test performance (83.3%), demonstrating effective transfer from ImageNet to CIFAR-100. However, the train/val gap (80% vs 69%) indicates some overfitting, and the drop to 75% on OOD data suggests the model's robustness to distribution shifts is still limited even with AugMix. Other hyperparameter configurations were explored, but OOD accuracy consistently stayed in the 69–71% range, making this configuration the best found.
 
---
 
## Part 3: Fine-Tuned Transformer
 
**Goal:** Fine-tune a large-scale pretrained Vision Transformer on CIFAR-100.
 
Three transformer architectures were explored: DeiT, Swin Transformer, and DINOv2 (all small variants, chosen for memory efficiency).
 
### Transformer Comparison
 
| Model | Train Acc | Val Acc | Clean Test | OOD Test |
|-------|-----------|---------|------------|----------|
| DeiT-small | 65% | 87% | — | 75.5% |
| Swin-small | 66% | 66% | 71% | 65.8% |
| **DINOv2-small (+ TTA)** | **65%** | **87%** | **91%** | **80.3%** |
 
### DINOv2 — Best Model
 
DINOv2 is a self-supervised vision transformer pretrained on large-scale image data. A two-stage fine-tuning strategy was used:
 
**Stage 1 — Head-only (5 epochs):** Backbone frozen, only the classification head trained. This allows the model to adapt its decision layer to CIFAR-100 without disrupting pretrained representations.
 
**Stage 2 — Full fine-tuning (45 epochs):** Entire model unfrozen with layer-wise learning rates — higher LR for the classification head, lower LR for transformer blocks and patch embeddings — to preserve pretrained features while enabling task-specific adaptation.
 
| Hyperparameter | Value |
|---------------|-------|
| Optimizer | AdamW |
| Weight decay | 0.05 |
| LR schedule | Linear warmup + cosine annealing |
| Input size | 224×224 (resized from 32×32) |
 
Test-Time Augmentation (TTA) was applied at inference, averaging predictions over multiple augmented views, yielding a further improvement from 79.9% to **80.3%** OOD accuracy.
 
### DeiT vs Swin
 
DeiT outperformed Swin under the same training setup. Swin was also tested with a more conservative augmentation pipeline to check whether aggressive augmentations were hurting performance — but this reduced OOD accuracy from DeiT's 75.5% to 65.8%, confirming that stronger augmentation is beneficial for robustness.
 
### Class-Level Analysis
 
**Best performing (Test Set):** Sunflower (82), Butterfly (14), Rabbit (75) — all have highly distinctive visual features rarely shared with other CIFAR-100 classes. Rabbit performed well even on a heavily distorted test image, reflecting DINOv2's robust shape representations from self-supervised pretraining.
 
**Best performing (Validation Set):** Goldfish (1), Keyboard (29), Dinosaur (39) — all correctly classified despite heavy augmentation (AugMix, Gaussian blur, random rotation), indicating robust learned representations for visually distinctive categories.
 
**Worst performing (Test Set):** Tree (47), Girl (98), Candle (10) — Tree is visually ambiguous and easily confused with other greenery. Girl is difficult due to high visual similarity with other human classes in CIFAR-100 (boy, man, woman). These failures suggest the model needs a better fine-tuning strategy for visually similar superclasses.
 
**Worst performing (Validation Set):** Boy (11), Plant (96), Tree (47) — Boy suffered from Random Erasing removing key facial features. Plant and Tree share high visual similarity, further distorted by heavy augmentation. Tree appearing in both worst-performing sets confirms it is a consistently difficult class.
 
---
 
## Conclusion
 
| Approach | OOD Test Accuracy | Rank |
|----------|------------------|------|
| DINOv2 + TTA | 80.3% | 🥇 1st |
| Fine-tuned ResNet50 | 75% | 🥈 2nd |
| Custom CNN (scratch) | 64.9% | 🥉 3rd |
 
The transformer model with self-supervised pretraining (DINOv2) achieved the strongest performance across both clean and OOD test sets. ResNet50 came second, while the scratch-built CNN — despite extensive tuning over 150 epochs — achieved moderate OOD robustness at 64.9%. This progression highlights the advantage of pretraining and transformer architectures in learning transferable, robust representations, particularly under distribution shift.
 
---
 
## Setup
 
Notebooks were trained on Kaggle (T4 GPU). To replicate, upload to Kaggle or Colab, enable GPU, and run all cells: CIFAR-100 and OOD test data download automatically.
 
---
 
## Key Design Decisions
 
- **No BN weight decay:** Weight decay excluded from BatchNorm and bias parameters across all models
- **Drop path (Part 1):** Stochastic depth applied at the residual branch level, increasing from 0.05 to 0.2 with depth
- **AugMix:** Primary augmentation strategy across all parts for OOD robustness
- **Label smoothing:** Prevents overconfident predictions, improves calibration
- **Layer-wise LR (Part 3):** Higher LR for head, lower for backbone during full fine-tuning
- **TTA at inference:** Averaging over multiple augmented views for a ~0.4% OOD accuracy boost
