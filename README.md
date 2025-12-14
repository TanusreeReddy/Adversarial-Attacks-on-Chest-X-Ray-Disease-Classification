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

