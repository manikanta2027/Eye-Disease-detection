# 👁️ Retinal Eye Disease Classification System

A deep learning-based image classification system that detects and classifies **10 types of retinal eye diseases** from fundus images using a custom **ResNet9 architecture** built in PyTorch.

---

## 📌 Project Overview

This system takes a retinal fundus image as input and classifies it into one of 10 disease categories, assisting in early diagnosis of eye conditions. The model is trained on the **Mendeley Eye Disease Image Dataset** containing ~16,140 augmented images.

---

## 🔍 Disease Classes

| # | Class |
|---|-------|
| 1 | Central Serous Chorioretinopathy |
| 2 | Diabetic Retinopathy |
| 3 | Disc Edema |
| 4 | Glaucoma |
| 5 | Healthy |
| 6 | Macular Scar |
| 7 | Myopia |
| 8 | Pterygium |
| 9 | Retinal Detachment |
| 10 | Retinitis Pigmentosa |

---

## 🗂️ Dataset

- **Source:** Mendeley Eye Disease Image Dataset
- **Total Images:** ~16,140
- **Image Type:** Color Fundus / Retinal Images
- **Input Size:** 128 × 128 pixels (RGB)
- **Split:**
  - Train: 70% (~11,298 images)
  - Validation: 15% (~2,421 images)
  - Test: 15% (~2,421 images)

---

## 🏗️ Model Architecture — ResNet9

A custom lightweight ResNet9 built from scratch with residual connections.

```
Input (3 × 128 × 128)
      │
   conv1 → (64 × 128 × 128)
      │
   conv2 → (128 × 64 × 64)
      │
   res1  → (128 × 64 × 64)  [Residual Block]
      │
   conv3 → (256 × 32 × 32)
      │
   conv4 → (512 × 16 × 16)
      │
   res2  → (512 × 16 × 16)  [Residual Block]
      │
   MaxPool(4) → Flatten → Dropout(0.2) → Linear(8192 → 10)
      │
  Output (10 classes)
```

Each `conv_block` consists of:
```
Conv2d → BatchNorm2d → ReLU (→ MaxPool2d if pool=True)
```

---

## ⚙️ Training Configuration

| Parameter | Value |
|-----------|-------|
| Optimizer | Adam |
| LR Scheduler | OneCycleLR |
| Max Learning Rate | 0.01 |
| Epochs | 14 |
| Batch Size | 64 |
| Gradient Clipping | 0.1 |
| Weight Decay | 1e-4 |
| Loss Function | Cross Entropy |
| Dropout | 0.2 |

---

## 🚀 Performance Improvement Techniques

- **Residual Connections** — prevent vanishing gradients
- **Batch Normalization** — stabilizes training, reduces covariate shift
- **Dropout (0.2)** — prevents overfitting
- **OneCycleLR Scheduler** — dynamic learning rate for faster convergence
- **Adam Optimizer** — adaptive learning rates
- **Gradient Clipping** — prevents exploding gradients
- **Weight Decay (L2 Regularization)** — reduces overfitting

---

## 🧩 System Components

1. **Data Pipeline** — Image loading, splitting, normalization, batch loading with GPU support
2. **Model Architecture** — Custom ResNet9 with conv blocks and residual connections
3. **Training Engine** — `fit_one_cycle()` with OneCycleLR, `fit()` for simple training
4. **Evaluation Module** — `evaluate()`, `accuracy()`, per-epoch logging
5. **Visualization Module** — Accuracy curves, loss curves, learning rate plots, batch preview

---

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/eye-disease-classification.git
cd eye-disease-classification

# Install dependencies
pip install torch torchvision matplotlib numpy opendatasets
```

---

## 🖥️ Usage

```python
# Load the trained model
model = ResNet9(in_channels=3, num_classes=10)
model.load_state_dict(torch.load("final.pth"))
model.eval()

# Predict on a single image
image = transform(PIL.Image.open("retina.jpg")).unsqueeze(0)
output = model(image)
_, predicted = torch.max(output, dim=1)
print(dataset.classes[predicted.item()])
```

---

## 📊 Visualization

```python
# Plot training accuracy
plot_Accuracies(history)

# Plot training vs validation loss
plot_losses(history)

# Plot learning rate schedule
plot_lrs(history)
```

---

## 💾 Saving & Loading Model

```python
# Save
torch.save(model.state_dict(), "final.pth")

# Load
model.load_state_dict(torch.load("final.pth"))
```

---

## 🛠️ Tech Stack

- **Language:** Python 3.x
- **Framework:** PyTorch
- **Libraries:** torchvision, matplotlib, numpy, opendatasets
- **Hardware:** CUDA-enabled GPU (recommended)

---

## 📁 Project Structure

```
eye-disease-classification/
│
├── eye-disease-image-dataset-mendeley/
│   └── Augmented Dataset/
│       ├── Central Serous Chorioretinopathy_Color Fundus/
│       ├── Diabetic Retinopathy/
│       └── ... (10 classes)
│
├── model.ipynb          # Main training notebook
├── final.pth            # Saved model weights
└── README.md
```

---

## 👨‍💻 Author

> Built as part of a deep learning project for automated retinal disease diagnosis.

---

## 📄 License

This project is for educational and research purposes only.
