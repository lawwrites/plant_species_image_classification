# Plant Species Image Classification Using Convolutional Neural Networks

## Research Overview

This repository documents a supervised deep learning study investigating the feasibility of automated plant species classification from RGB imagery using a Convolutional Neural Network (CNN).

The objective of this work is to evaluate whether a moderately sized CNN, trained on standardized 128×128 RGB plant images, can achieve robust multi-class classification across 12 morphologically distinct plant species.

The broader motivation for this research includes:

* Supporting ecological monitoring workflows
* Assisting early-stage crop and weed identification
* Reducing manual annotation burden in agricultural research settings

---

## Dataset Description

The dataset consists of:

* **4,750 RGB images**
* Image resolution: **128 × 128 pixels**
* **12 plant species classes**

Class labels were encoded numerically to support multi-class optimization using CrossEntropyLoss.

To preserve class balance across evaluation stages, stratified sampling was applied:

* Training set: 3,800 images (80%)
* Validation set: 475 images (10%)
* Test set: 475 images (10%)

Class distribution analysis confirmed moderate imbalance across species, which was addressed via stratified splitting rather than resampling.

---

## Preprocessing and Data Pipeline

All preprocessing was implemented using a custom PyTorch Dataset class to maintain full control over the transformation pipeline.

### Image Standardization

1. Resize to 128 × 128 pixels
2. Convert to floating-point tensor (C, H, W format)
3. Normalize using training-set channel statistics

Training set statistics:

* Mean: [0.1403, 0.1614, 0.1717]
* Std:  [0.1335, 0.1478, 0.1652]

Normalization ensures stable gradient propagation and improves convergence reliability.

### Data Augmentation (Training Only)

To improve generalization:

* Random horizontal flipping (p = 0.5)
* Random rotation within ±20°

Augmentation was applied exclusively to the training set to prevent information leakage.

---

## Model Architecture

A custom Convolutional Neural Network was designed with three convolutional blocks followed by a fully connected classifier.

### Feature Extraction Layers

* Conv2D (3 → 32), kernel 3×3, padding=1

* ReLU activation

* MaxPooling (2×2)

* Conv2D (32 → 64)

* ReLU

* MaxPooling

* Conv2D (64 → 128)

* ReLU

* MaxPooling

After three pooling operations, spatial resolution is reduced to 16×16 with 128 channels.

### Classification Head

* Flatten (128 × 16 × 16 → 32,768)
* Fully Connected (32,768 → 256)
* ReLU
* Dropout (p = 0.3)
* Fully Connected (256 → 12)

Total trainable parameters: **8,485,196**

The architecture balances representational capacity with computational efficiency suitable for CPU training.

---

## Optimization Strategy

Loss Function:

* CrossEntropyLoss (multi-class classification)

Optimizer:

* Adam
* Learning rate: 3 × 10⁻⁴
* Weight decay: 1 × 10⁻⁴

Training duration:

* Maximum 10 epochs
* Early stopping with patience = 3 based on validation loss

The best-performing model (lowest validation loss) was checkpointed and restored for final evaluation.

---

## Experimental Results

### Final Training Metrics

* Loss: 0.425
* Accuracy: 85.3%

### Final Validation Metrics

* Loss: 0.404
* Accuracy: 87.2%
* Macro F1-score: 0.851

The macro F1-score indicates balanced predictive performance across classes despite moderate class imbalance.

### Confusion Matrix Analysis

Confusion matrix inspection revealed:

* Strong diagonal dominance across most species
* Minor misclassification between morphologically similar species

These results suggest that the learned convolutional filters successfully capture discriminative structural and color-based features.

---

## Model Persistence

The trained model weights are stored as:

`cnn_flower_model.pth`

This enables reproducibility and downstream experimentation without retraining.

---

## Research Contributions Demonstrated

* End-to-end deep learning pipeline implementation in PyTorch
* Controlled image normalization using dataset-specific statistics
* Custom data augmentation without reliance on high-level wrappers
* CNN architecture design for moderate-resolution ecological imagery
* Early stopping implementation to mitigate overfitting
* Multi-metric evaluation including macro F1 and confusion matrix analysis

---

## Future Work

Potential extensions include:

* Transfer learning using pretrained backbones (e.g., ResNet)
* Class imbalance mitigation via weighted loss functions
* Integration of Grad-CAM for model interpretability
* Deployment for field-based inference

---

This repository represents an applied investigation into automated plant species recognition using convolutional neural networks, with emphasis on reproducibility, controlled experimentation, and measurable performance evaluation.
