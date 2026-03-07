# Underwater Scene Clustering for Structural Health Monitoring

> Code for the paper:  
> **"Unsupervised Scene-Based Clustering for Underwater Structural Health Monitoring: Enhancing Image Categorization and Feature Matching"**  

---

## Overview

Unsupervised scene-based clustering framework that automatically organizes underwater inspection images by environmental conditions (visibility, color, turbidity) without labeled training data. Enhances downstream tasks such as feature matching, image alignment, and 3D reconstruction for underwater structural health monitoring.

## Requirements

- Python 3.9+
- See `requirements.txt` for dependencies

## Setup

```bash  
pip install -r python/requirements.txt
```

## Usage
Place images in `data/input/<dataset_name>/`, update `python/config.py` if needed, then run:

```bash  
python python/main.py
```
Outputs are saved to `data/output/`.

## Datasets
See [DATASETS.md](DATASETS.md)
