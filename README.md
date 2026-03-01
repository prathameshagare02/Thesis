# Automated Visual Corrosion Analysis for Spray Experiments

**Author:** Prathamesh Agare  
**Affiliation:** FAU Erlangen-Nürnberg | Schaeffler Technologies AG & Co. KG
**Year:** 2024–2025  

---

## 🎯 Overview

This repository hosts the code and research materials for the **AI-based corrosion detection and progression prediction** project.  
The goal is to build an automated, data-driven pipeline that identifies, classifies, and tracks corrosion formation and growth over time from image data of spray experiments.

---

## 🧠 Thesis Question

Can an AI model identify early-stage corrosion and predict its growth path using annotated image datasets?

---

## 🧩 Work Packages

### **WP1 – Image Normalization**
- Extract and align surface regions from time-lapse corrosion images.  
- Handle camera perspective variations through **homography-based alignment**.  
- Apply **self-supervised feature extraction (e.g., DINO, SimCLR)** for consistent surface representation prior to segmentation.

---

### **WP2 – Unsupervised Corrosion Segmentation**
- Instead of manually labeled training data, this stage focuses on **clustering corrosion regions** based on texture, color, and learned visual embeddings.  
- Techniques explored:
  - **Autoencoder-based latent clustering** (VAE, DeepCluster)
  - **Contrastive vision transformers** (e.g., DINO-ViT)
  - **Region-based refinement** using **watershed** and **graph-based segmentation**  
- Objective: Achieve corrosion detection without relying on human annotation while preserving meaningful physical boundaries.

**Related Work**
- “Unsupervised Image Segmentation by Leveraging Representation Learning”
- “Self-supervised Vision Transformers for Industrial Surface Anomaly Detection”
- “Fusion of Unsupervised and Semantic Segmentation for Corrosion Mapping”

---

### **WP3 – Temporal Tracking and Progression Prediction**
- Use **unsupervised feature consistency** to track corrosion growth per pixel or per cluster across sequential images.  
- Combine **optical flow** with **embedding-space matching** to detect subtle temporal changes.  
- Estimate **growth rate maps** and visualize corrosion kinetics as temporal heatmaps.

---

## 🧠 Why Unsupervised Segmentation?
Labeling corrosion data is both subjective and time-consuming.  
Unsupervised segmentation:
- Removes dependency on labeled datasets.  
- Allows the model to generalize across materials, lighting, and corrosion stages.  
- Enables integration with self-supervised representations, making it suitable for industrial-scale applications.

---

## 🧪 Proposed Methodology (Updated)
**Input:** Sequential corrosion images  
**Preprocessing:** Denoising, normalization, geometric alignment  
**Modeling:**
- **Feature Extraction:** Self-supervised ViT / DINO embeddings  
- **Segmentation:** Clustering in embedding space + morphological refinement  
- **Temporal Prediction:** Optical flow or LSTM tracking of cluster evolution  

**Output:**
- Corrosion segmentation masks  
- Cluster-wise growth rates  
- Unsupervised corrosion severity mapping  


