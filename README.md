# SCALE × ODYSSEY — Mid-Term Evaluation Report
**Submitted by:** Dhyey Buch | IIT Gandhinagar CSE Year 1  
**Date:** June 2026 | **Status:** Work in Progress

---

## 1. Dataset Sources

Two datasets were combined after determining that no single source covered all five required classes adequately.

| Source | Classes Extracted | Images | Origin |
|---|---|---|---|
| Galaxy Zoo Classification (Kaggle) | Spiral Galaxy, Elliptical Galaxy | ~24,890 | Crowd-sourced, volunteer-labelled morphology |
| SpaceNet FLARE (Kaggle) | Nebula, Planetary Object, Star Cluster | ~5,933 | MBZUAI research dataset, ESA SPAICE 2024 |
| **Total** | **5 classes** | **~30,823** | |

**Key preprocessing decisions:**
- Galaxy Zoo's 3 elliptical subclasses (round smooth, in-between smooth, cigar-shaped) merged into single Elliptical class
- Edge-on galaxies discarded entirely from the dataset
- All images resized to 224×224 JPEG at preprocessing time to fit Kaggle's 19.5GB disk quota
- WeightedRandomSampler used during training to handle class imbalance (elliptical ~17,000 vs nebula ~1,192)

---

## 2. Model Architecture

**Base model:** ResNet-50 pretrained on ImageNet-1K

**Modifications:**
- Classification head replaced: `2048 → 512 (ReLU, Dropout 0.4) → 5`
- Full model fine-tuned with differential learning rates per layer group

| Layer Group | Learning Rate 
|---|---|---|
| conv1, bn1, layer1 | 1e-6 
| layer2 | 1e-6
| layer3 | 5e-5
| layer4 | 1e-4 
| fc (head) | 5e-4 

**Training configuration:**
- Optimiser: AdamW (weight decay 1e-4)
- Scheduler: CosineAnnealingLR (T_max=25, η_min=1e-7)
- Gradient clipping: max norm 1.0 (prevents explosive gradients in early layers)
- Early stopping: patience 7 epochs
- Batch size: 64

**Augmentation (training only):**
RandomHorizontalFlip, RandomVerticalFlip, RandomRotation(180), RandomGrayscale(p=0.2), ColorJitter
---

## 3. Final Validation Metrics

**Overall Accuracy: 96.82%**

| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| Elliptical Galaxy | 0.99 | 0.98 | 0.99 | 2,552 |
| Nebula | 0.83 | 0.88 | 0.85 | 171 |
| Planetary Object | 0.93 | 0.97 | 0.95 | 230 |
| Spiral Galaxy | 0.95 | 0.99 | 0.97 | 1,144 |
| Star Cluster | 0.95 | 0.91 | 0.93 | 526 |
| **Macro Average** | **0.93** | **0.94** | **0.94** | **4,623** |
| **Weighted Average** | **0.97** | **0.97** | **0.97** | **4,623** |

**Accuracy progression across experiments:**

| Experiment | Val Accuracy | Macro F1 | Key Change |
|---|---|---|---|
| ResNet18, layer4+fc | 88.16% | 0.87 | Baseline |
| ResNet18, layer3+4+fc | 90.35% | 0.89 | Unfreeze layer3 |
| ResNet50, layer3+4+fc | 92.98% | 0.89 | Deeper backbone |
| ResNet50, full fine-tune | **96.82%** | **0.94** | Differential LR + deeper head |

---

## 4. Confusion Matrix

*(To be added — test set evaluation pending final model checkpoint)*

**Known confusion patterns from validation analysis:**
- Elliptical ↔ Spiral: 27 ellipticals predicted as spiral (visually similar edge cases)
- Nebula ↔ Star Cluster: primary confusion pair — diffuse structures share visual features
- Star Cluster ↔ Planetary: 35 misclassifications — globular clusters and planets share spherical morphology
- Galaxy ↔ Non-galaxy boundary: zero cross-confusion — model perfectly separates these domains

---

## 5. Inference Time

~660ms on Kaggle GPU

**To measure:**
```python
import time
start = time.time()
predict(image, model, device)
elapsed = (time.time() - start) * 1000
print(f"Inference time: {elapsed:.1f}ms")
```

---

## 6. Setup Instructions

**Requirements:** Python 3.10+, CUDA-compatible GPU recommended

```bash
# 1. Clone repository
git clone https://github.com/dhyeybuch/scale-odyssey.git
cd scale-odyssey

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download datasets (Kaggle CLI)
kaggle datasets download anjosut/galaxy-zoo-classification
kaggle datasets download razaimam45/spacenet-an-optimally-distributed-astronomy-data

# 4. Build final dataset
python build_dataset.py \
    --galaxy_zoo path/to/galaxy_zoo \
    --spacenet   path/to/spacenet \
    --output_dir data/final_dataset

# 5. Run inference on a single image
python predict.py --image your_image.jpg

# 6. Launch interactive demo
python app.py
# Opens at http://localhost:7860
```

**Live demo (no setup required):**  
`[HuggingFace Spaces URL — to be added on deployment]`

---

## 7. Training Script

**Full training run:**
```bash
python train.py
```

Trains ResNet-50 with differential learning rates on the merged dataset. Saves best checkpoint to `outputs/models/best_model.pth` based on validation loss. Prints classification report every 5 epochs. Early stopping with patience=7.

**Key files:**

| File | Purpose |
|---|---|
| `config.py` | All hyperparameters and paths — single source of truth |
| `build_dataset.py` | One-time dataset construction from raw sources |
| `dataset.py` | DataLoaders with WeightedRandomSampler |
| `model.py` | ResNet-50 definition, optimizer, checkpoint loading |
| `train.py` | Full training loop with CosineAnnealingLR |
| `predict.py` | Single image inference via CLI |
| `gradcam.py` | GradCAM visualisation and analysis |
| `app.py` | Gradio web demo |
