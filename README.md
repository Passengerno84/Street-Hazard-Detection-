# Street-Hazard-Detection

# YOLO-MAML: Few-Shot Object Detection for Road Damage

This repository implements a **few-shot object detection pipeline** combining **YOLOv8** with **Model-Agnostic Meta-Learning (MAML)**.  
The goal is to enable road damage detection (e.g., cracks, potholes, and other corruptions) in **low-data scenarios**, where only a few annotated samples are available.

---

## 📌 Features
- **Custom few-shot sampler** for N-way K-shot tasks.
- **Meta-learning training loop (MAML)** on top of YOLOv8.
- **Support for multiple damage classes** (Crack, Pothole, Corruption).
- **Evaluation pipeline** with metrics: Precision, Recall, mAP@0.5, mAP@0.5:0.95.
- **Visualization** of predictions during testing.

---
