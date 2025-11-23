# Assignment 3: PyTorch Training Pipeline & Experimental Report

**Author:** Ionescu Emi  
**Date:** November 23, 2025  
**Course:** Advanced Topics in Neural Networks  

---

## 1. Introduction
This report details the implementation of a generic training pipeline using PyTorch. The pipeline is designed to be fully configurable, supporting multiple architectures, optimizers, and datasets, while integrating robust logging and scheduling mechanisms.

The experimental focus of this report is on the **CIFAR-100** dataset, utilizing both Transfer Learning (Pre-training) and training from scratch (No Pre-training) to meet the accuracy targets outlined in the assignment rubric.

---

## 2. Pipeline Architecture
The implemented pipeline satisfies the required feature set for 8 points:

* **Device Agnostic:** The code automatically detects available hardware (`cuda` vs `cpu`).
* **Dataset Support:** Configurable support for MNIST, CIFAR-10, CIFAR-100, and OxfordIIITPet. The pipeline utilizes `num_workers` and `pin_memory` for efficient data loading.
* **Models:** Integrates `timm` and `torchvision` to load models such as `resnet18`, `efficientnet`, and `wideresnet`.
* **Optimization:**
    * **Optimizers:** Support for SGD, Adam, AdamW, and a custom implementation of **SAM (Sharpness-Aware Minimization)**.
    * **Schedulers:** Integration of `ReduceLROnPlateau` and `StepLR` (MultiStepLR used in final run).
    * **Batch Scheduling:** Integrates a batch size scheduler.
* **Logging:** Full integration with **Weights & Biases (WandB)** for metric reporting and hyperparameter sweeping.
* **Early Stopping:** Implemented to monitor validation accuracy and save the best model weights.

---

## 3. Hyperparameter Sweep (Pre-trained Models)
A hyperparameter sweep was conducted using WandB to satisfy the requirement of identifying 8 configurations achieving over 70% accuracy on CIFAR-100.

**Sweep Parameters:**  
Per the requirement to describe varied parameters, the following were swept:

* **Models:** `resnet18`, `efficientnet_b0` (Pre-trained on ImageNet).
* **Optimizers:** `adamw`, `sam`.
* **Batch Size:** `64`, `96`.

**Results:**  
We successfully identified **8 distinct configurations** that exceeded the 70% threshold.

| Run ID | Model | Optimizer | Batch Size | Val Accuracy | Time (s) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `dtjhdjve` | EfficientNet_B0 | AdamW | 96 | **76.13%** | 145.9 |
| `870nhqhw` | EfficientNet_B0 | SAM | 96 | **75.04%** | 147.6 |
| `zxmixt61` | EfficientNet_B0 | AdamW | 96 | **74.70%** | 146.0 |
| `9fjpkviy` | ResNet18 | AdamW | 96 | **72.32%** | 85.2 |
| `22nuswnu` | ResNet18 | AdamW | 96 | **72.28%** | 85.4 |
| `uhm1yuyx` | ResNet18 | SAM | 64 | **71.66%** | 165.4 |
| `v1k3wsz8` | ResNet18 | AdamW | 96 | **71.33%** | 85.1 |
| `b5oro9oy` | ResNet18 | SAM | 96 | **70.40%** | 161.1 |

*Table 1: 8 Configurations achieving >70% accuracy on CIFAR-100.*
<img width="3160" height="1660" alt="W B Chart 11_23_2025, 11_49_02 PM" src="https://github.com/user-attachments/assets/1c2d87f8-d199-4e03-a480-0bfb65d31f2b" />

*Figure 1: Parallel coordinates plot from WandB showing the impact of Batch Size and Optimizer on Accuracy.*

---

## 4. Best Run: No Pre-training (80.05%)
To address the "No Pre-training" criteria, a dedicated training run was performed from scratch.

**Configuration:**
* **Model:** WideResNet-28-10
* **Optimizer:** SAM (Rho=0.05) with SGD base.
* **Augmentation:** Cutout, RandomCrop, RandomHorizontalFlip.
* **Scheduler:** MultiStepLR (Milestones: 100, 150).
* **Epochs:** 200

**Performance:**  
The model demonstrated steady convergence, with the SAM optimizer effectively smoothing the loss landscape in the final training phase.

* **Final Test Accuracy:** **80.05%** (EMA)  
* **Training Time:** 9.26 hours  

By achieving **80.05%**, this run satisfies the criteria for both 79% and 80% accuracy.


---

## 5. Comparison: No Pre-training vs. Pre-training
As requested, we compare the two approaches:

| Metric | No Pre-training (Best Run) | Pre-training (Best Sweep Run) |
| :--- | :--- | :--- |
| **Best Accuracy** | **80.05%** | 76.13% (Limited epochs) |
| **Convergence Speed** | Slow (~100 epochs to reach 70%) | Instant (< 1 epoch to reach 70%) |
| **Compute Cost** | High (~9.26 hours) | Low (~2.5 minutes) |
| **Architecture** | WideResNet-28-10 | EfficientNet_B0 |

**Motivation for Efficiency:**  
While pre-training is significantly more efficient in terms of time, the "No Pre-training" pipeline was optimized using **Mixed Precision (AMP)** and **JAX-based forward passes** (implemented as an optional flag). The JAX forward pass yielded an approximate **17% reduction** in wall-clock time compared to the standard PyTorch forward pass, demonstrating efficient resource usage.

---

## 6. Setup + How to Run
Per the assignment requirements, the full code and reproducible runs are available via the following Kaggle Notebook links:

* **Training Pipeline & Hyperparameter Sweep:**  
  https://www.kaggle.com/code/emi0011/notebook1b270dee9a  
  *Contains the full generic pipeline, the 8-config sweep, and the JAX/PyTorch comparison logic.*

* **Best Model Run (No Pre-training, 80.05%):**  
  https://www.kaggle.com/code/emi0011/notebookc20df812fe  
  *Contains the standalone reproduction of the WideResNet-28-10 + SAM configuration achieving >80% accuracy.*

---
