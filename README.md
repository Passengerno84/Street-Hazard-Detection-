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

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/Passengerno84/Street-Hazard-Detection-.git
cd Street-Hazard-Detection-
```

---
 
### 2. Create a virtual environment (recommended):
```bash
python -m venv venv
```

---

### 3. Activate the environment:
- **Windows**
```bash
venv\Scripts\activate ```

- **Linux / macOS**
```bash
source venv/bin/activate ```

---

### 4. Install dependencies:
```bash
pip install -r requirements.txt ```

---

## Usage

Run the main script to start training or testing:

- **Train the model:**
```bash
python main.py --train

- **Test the model:**
```bash
python main.py --test

---

## ⚠️ Limitations
- Current results show very low performance on evaluation datasets.
- Training stability is inconsistent in few-shot settings.
- NMS filtering discards most predictions due to low confidence.

---

## ⚠️ Limitations
- Current results show very low performance on evaluation datasets.
- Training stability is inconsistent in few-shot settings.
- NMS filtering discards most predictions due to low confidence.

---

## 🔮 Future Improvements
- Explore data augmentation and synthetic sample generation.
- Explore data augmentation and synthetic sample generation
- Implement better meta-learning strategies beyond vanilla MAML (e.g., ProtoNet, Reptile).

---

