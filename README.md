# Diabetes Indicators ML + CNN (BRFSS 2015 + Vegetable Images)

This notebook contains two parts:
1) A machine learning analysis of diabetes risk using BRFSS 2015 health indicators (EDA, PCA, clustering, and supervised classification).
2) A CNN-based image classification workflow using a Vegetable Images dataset.

## Part 1 — Diabetes Health Indicators (BRFSS 2015)
**Workflow**
- Data loading and preprocessing
- Correlation analysis + distribution plots
- Class balance inspection
- PCA (principal components + explained variance)
- Clustering: K-Means, DBSCAN, Agglomerative, Gaussian Mixture
- Clustering evaluation using NMI and ARI

**Supervised Models**
- Logistic Regression (GridSearchCV over C and class_weight; AUC-ROC scoring)
- Random Forest (GridSearchCV over depth/trees/split; AUC-ROC scoring)
- Random Forest (class_weight=balanced)

**Metrics reported**
Accuracy, Precision, Recall, F1-Score, and AUC-ROC.

## Part 2 — CNN Image Classification
- Loads train/validation/test splits from a Vegetable Images dataset
- Builds and trains a CNN model for multi-class classification

## Data
Raw datasets are not included in this repository. If rerunning:
- BRFSS CSV is expected under `archive/`
- Vegetable Images data is expected under `Vegetable Images/`
Update paths in the notebook if your local structure differs.

## How to run
```bash
jupyter notebook diabetes.ipynb
