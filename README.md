# Adversarial Attacks on Chest X-Ray Disease Classification

This repository presents an experimental study on the vulnerability of deep learning models for medical image analysis to adversarial attacks. Specifically, we evaluate the impact of the **Fast Gradient Sign Method (FGSM)** on a convolutional neural network (CNN) trained to classify chest X-ray images as **PNEUMONIA** or **NORMAL**.

The project demonstrates that even small, visually imperceptible perturbations can significantly degrade the performance of medical image classifiers, raising serious concerns for real-world clinical deployment.

---

## 📌 Project Overview

- **Task:** Binary classification (PNEUMONIA vs NORMAL)
- **Dataset:** Chest X-Ray Images (Pneumonia)
- **Model:** Custom CNN
- **Attack Method:** Fast Gradient Sign Method (FGSM)
- **Framework:** PyTorch
- **Platform:** Google Colab (GPU)

---

## 📂 Dataset

The project uses the publicly available **Chest X-Ray Images (Pneumonia)** dataset, organized into training, validation, and test splits:


### Test Set Distribution

| Class      | Count |
|------------|-------|
| NORMAL     | 234   |
| PNEUMONIA  | 390   |

Images are resized to **224×224**, normalized to the range **[-1, 1]**, and converted to three-channel tensors.

---

## 🧠 Model Architecture

The CNN consists of three convolutional blocks followed by fully connected layers:

- Conv (32 filters) → ReLU → MaxPool
- Conv (64 filters) → ReLU → MaxPool
- Conv (128 filters) → ReLU → MaxPool
- Fully Connected (256 units) + Dropout
- Output layer with Sigmoid activation

The model outputs a probability score thresholded at 0.5 for binary classification.

---

## ⚙️ Training Configuration

- **Optimizer:** Adam  
- **Learning Rate:** 1e-4  
- **Loss Function:** Binary Cross-Entropy Loss  
- **Batch Size:** 32  
- **Epochs:** 5  
- **Data Augmentation:** Random horizontal flip (training only)

---

## ⚔️ Adversarial Attack: FGSM

The Fast Gradient Sign Method generates adversarial examples by perturbing the input image in the direction of the gradient of the loss with respect to the input:

\[
x_{adv} = x + \epsilon \cdot \text{sign}(\nabla_x J(\theta, x, y))
\]

We evaluate the model under different values of **ε (epsilon)** to study robustness degradation.

---

## 📊 Results

### Accuracy vs FGSM Attack Strength

| ε (Epsilon) | Test Accuracy |
|------------|---------------|
| 0.00       | 75.2%         |
| 0.01       | 67.6%         |
| 0.03       | 57.5%         |
| 0.05       | 45.5%         |
| 0.10       | 14.6%         |

The results show a **monotonic decline in accuracy** as attack strength increases, confirming the model’s vulnerability to adversarial perturbations.

## 🔍 Observations

- The model shows **high sensitivity** to pneumonia cases but **low specificity** for normal cases.
- Even small perturbations (ε = 0.01) significantly degrade performance.
- Strong attacks render the model nearly unusable.
- Adversarial images appear visually similar to clean images but cause drastic prediction changes.

---

## 🚀 How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/Adversarial-Attacks-ChestXray.git
