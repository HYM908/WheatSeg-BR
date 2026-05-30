# WheatSeg-BR
This repository provides the core code implementation of WheatSeg-BR, a boundary-aware semantic segmentation method for field wheat point clouds that supports the joint segmentation of three organ categories: stem, leaf, and soil.
## 📌 Overview

This repository provides the implementation of **WheatSeg-BR**, a boundary-aware semantic segmentation framework for field wheat point clouds.

The goal of this work is to segment **stems**, **leaves**, and **soil** from real field wheat point clouds. Compared with greenhouse or single-plant scenarios, field wheat point clouds are much more challenging due to dense plant growth, severe occlusion, uneven point density, incomplete reconstruction, slender stems, and ambiguous organ boundaries.

WheatSeg-BR is designed to improve semantic segmentation performance in complex field scenarios by enhancing the representation of sparse slender structures and ambiguous stem–leaf and stem–soil boundaries.

---

## ✨ Key Features

- 🌱 **Field wheat point cloud segmentation**  
  Designed for real field population-scale wheat point clouds rather than controlled single-plant data.

- 🌾 **Stem–leaf–soil semantic segmentation**  
  Supports organ-level segmentation of wheat stems, leaves, and soil background.

- 🔍 **Boundary-aware modeling**  
  Improves the discrimination of ambiguous stem–leaf and stem–soil boundary regions.

- 🧩 **Improved model components**  
  The `models/` folder contains the main improved modules used in WheatSeg-BR.

- 📂 **Example data included**  
  The `data/` folder provides three sample point cloud files for testing and demonstration.

---

## 🗂️ Repository Structure

```text
WheatSeg-BR/
│
├── models/                 # Improved modules and network components of WheatSeg-BR
│   ├── ...
│
├── data/                   # Example point cloud data
│   ├── data1.*
│   ├── data2.*
│   └── data3.*
│
├── README.md               # Project description
└── requirements.txt        # Python dependencies
