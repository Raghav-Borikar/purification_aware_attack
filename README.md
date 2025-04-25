# 🔒 Purification-Aware Attack (PAA)

This repository contains the implementation of our project: **"Defeating CLIPure-Cos with Purification-Aware Attack"**. introduces a novel **Purification-Aware Attack (PAA)** that circumvents CLIPure-Cos with high success across CIFAR-10, CIFAR-100, and ImageNet datasets.

## 🧠 Overview

- **Defense Replicated**: CLIPure-Cos (a purification-based defense using cosine similarity in CLIP latent space)
- **Novel Contribution**: A differentiable, momentum-aware adversarial attack that anticipates and manipulates the purification process.
- **Results**:
  - CLIPure-Cos replication shows ~72% clean accuracy and ~67% robust accuracy.
  - PAA achieves over **90% attack success** against CLIPure-Cos.

## 🗂 Project Structure
├── main.py # Entry point for running experiments

├── requirements.txt # Python dependencies

├── config.py # Configuration file

├── clipuremodel.py # CLIP model & utilities

├── paa_attack.py # Purification aware attack implementation

├── evaluation.py # Integration code

└── utils.py # Preprocessing, logging, evaluation utilities

---

## 🚀 How to Run

### 1. Clone this Repository

```bash
git clone https://github.com/yourusername/purification_aware_attack.git
cd clipure-paa
```

### 2. Install Dependencies

Make sure you have Python 3.8+ and pip installed.

```bash
pip install -r requirements.txt
pip install git+https://github.com/bypanda/torchattack.git
```

### 3. Dataset Setup
   
CIFAR-10 & CIFAR-100 will be automatically downloaded.

ImageNet: You must manually download and specify its location in config.py.

#### Example config.py snippet
```bash
imagenet_path: /path/to/imagenet/
```

### 4. Run Experiments
Replace <dataset> with cifar10, cifar100, or imagenet.

```bash
python main.py --dataset <dataset>
```

```bibtex
@misc{clipurepaa2025,
  author = {Raghav Borikar and Byomakesh Panda},
  title = {Defeating CLIPure-Cos with Purification-Aware Attack},
  year = {2025},
  note = {Project Report, CS607 - Adversarial Machine Learning}
}
```
