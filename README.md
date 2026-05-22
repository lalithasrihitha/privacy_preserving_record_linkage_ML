# privacy_preserving_record_linkage_ML
Comparative evaluation of machine learning models for privacy-preserving record linkage using real-world healthcare data, sampling strategies, and statistical validation techniques.
# Privacy-Preserving Record Linkage Using Machine Learning for Real-World Healthcare Data

## Overview

This project presents a comparative evaluation of supervised machine learning models for Privacy-Preserving Record Linkage (PPRL) using real-world healthcare data represented through binary agreement features.

The project was completed as part of a practicum internship at the Regenstrief Institute in collaboration with Indiana University Luddy School of Informatics, Computing, and Engineering.

The primary goal of this work was to evaluate how effectively different machine learning approaches can identify matching healthcare records while preserving patient privacy and minimizing exposure of sensitive patient identifiers.

---

# Background

Healthcare information is frequently fragmented across multiple hospitals, clinics, healthcare systems, and electronic health record (EHR) platforms. Patients often receive care from different healthcare organizations, resulting in incomplete longitudinal patient records.

Traditional record linkage approaches commonly rely on personally identifiable information (PII), including:
- Patient names
- Date of birth
- Addresses
- Phone numbers
- Social Security Numbers

Although effective, direct use of sensitive identifiers introduces:
- Privacy concerns
- Security risks
- HIPAA compliance challenges
- Data-sharing limitations

Privacy-Preserving Record Linkage (PPRL) addresses this challenge by enabling secure patient matching using transformed or encoded representations instead of raw identifiers.

This project explores how machine learning models perform on privacy-preserved healthcare linkage tasks using real-world healthcare data.

---

# Objectives

The objectives of this project were to:

- Develop a privacy-preserving healthcare record linkage workflow
- Generate binary agreement features from transformed healthcare identifiers
- Compare multiple supervised machine learning models
- Evaluate the impact of class imbalance handling techniques
- Analyze precision–recall trade-offs in healthcare linkage
- Perform statistical validation of model performance differences
- Investigate whether increasing model complexity improves linkage performance

---

# Dataset

The study used 10,000 labeled healthcare record pairs derived from real-world healthcare data.

Each record pair was represented using binary agreement features:
- `1` = agreement between compared fields
- `0` = disagreement between compared fields

The dataset included:
- Matching record pairs
- Non-matching record pairs
- Class imbalance between matches and non-matches

To preserve patient privacy:
- raw identifiers were not directly exposed
- transformed representations were used instead

Binary agreement features were generated from transformed healthcare identifiers by comparing corresponding fields between record pairs. Agreement between compared fields was encoded as `1`, while disagreement was encoded as `0`.

The dataset underwent preprocessing and transformation prior to machine learning experimentation to support privacy-preserving record linkage workflows.

---

# Dataset Availability

The original dataset used in this project is not publicly available due to healthcare privacy, institutional regulations, and data protection restrictions.

The study was conducted using privacy-preserving transformed healthcare data derived from real-world healthcare record pairs during a practicum internship at the Regenstrief Institute.

To maintain compliance with healthcare privacy and security standards:
- raw patient identifiers were not publicly shared
- protected health information (PHI) was not exposed
- original healthcare datasets are excluded from this repository

This repository contains:
- preprocessing workflows
- machine learning pipelines
- statistical evaluation scripts
- sampling strategy experiments
- reproducible model evaluation code

developed for privacy-preserving healthcare record linkage research.

No identifiable patient information is included in this repository.

---

# Machine Learning Models Evaluated

The following supervised machine learning models were comparatively evaluated:

## Logistic Regression
Used as a baseline interpretable classification model for structured binary agreement data.

## Support Vector Machine (SVM)
Evaluated for its ability to identify optimal decision boundaries between matching and non-matching record pairs.

## K-Nearest Neighbors (KNN)
Used to evaluate similarity-based classification performance within healthcare linkage tasks.

## Single-layer Neural Network (SLNN)
Evaluated to determine whether limited nonlinear learning improves classification performance.

## Multi-layer Neural Network (MLNN)
Used to investigate the effect of increased network complexity and deeper architectures.

## XGBoost
Evaluated as an ensemble-based gradient boosting model capable of learning complex feature interactions.

---

# Class Imbalance Handling

Because the dataset demonstrated class imbalance, multiple sampling strategies were evaluated:

## Original Dataset
Baseline performance without modifying class distribution.

## Upsampling
Minority class duplication to increase class balance.

## Downsampling
Reduction of majority class samples to create balanced class distributions.

## SMOTE (Synthetic Minority Oversampling Technique)
Synthetic generation of minority class samples using nearest-neighbor interpolation.

---

# Evaluation Metrics

The following evaluation metrics were used:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix
- Precision–Recall Analysis

Special attention was given to precision–recall trade-offs because healthcare record linkage requires balancing:
- False Positive Matches
- False Negative Matches

False matches may incorrectly combine records from different patients, while missed matches may fragment patient histories across systems.

---

# Statistical Validation

To ensure robust comparison between machine learning models, multiple statistical significance tests were performed:

## McNemar Test
Used to compare paired classification outcomes between models.

## Paired t-test
Used to compare mean performance differences across evaluation metrics.

## Wilcoxon Signed-Rank Test
Used as a non-parametric alternative for validating performance differences.

These analyses helped determine whether observed differences between models represented statistically meaningful improvements or random variation.

---

# Key Findings

- Single-layer Neural Network (SLNN) achieved the highest overall performance.
- Logistic Regression, SVM, and MLNN demonstrated statistically comparable performance.
- XGBoost showed comparatively lower recall across sampling strategies.
- KNN achieved high recall but lower precision due to increased false positives.
- Simpler interpretable models performed competitively with more complex architectures.
- Performance differences were influenced more strongly by feature representation and class imbalance than by increasing model complexity.

---

# Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Imbalanced-learn
- SciPy
- Matplotlib

---

# Repository Structure

```text
privacy-preserving-record-linkage-ml/
│
├── README.md
├── requirements.txt
├── poster/
├── paper/
├── scripts/
├── outputs/
└── data/
```

---

# Project Files

## Preprocessing and Feature Engineering

### `pprl_preprocess.py`
Handles preprocessing of healthcare record-pair data and generates binary agreement features and labels for machine learning workflows.

### `pprl_weights.py`
Implements weighted comparison logic and feature preparation for privacy-preserving record linkage experiments.

### `BASELINE.py`
Runs baseline machine learning experiments using the original dataset distribution.

### `BALANCED.py`
Implements balanced dataset workflows using sampling strategies to address class imbalance.

---

# Logistic Regression Files

### `logistic_regression.py`
Trains and evaluates Logistic Regression on the original dataset.

### `logistic_regression_upsample.py`
Evaluates Logistic Regression using upsampling-based class balancing.

### `logistic_regression_downsample.py`
Evaluates Logistic Regression using downsampling-based class balancing.

---

# Support Vector Machine Files

### `svm.py`
Implements Support Vector Machine classification on the original dataset.

### `svm_upsample.py`
Evaluates SVM performance using upsampled training data.

### `svm_downsample.py`
Evaluates SVM performance using downsampled training data.

---

# K-Nearest Neighbors Files

### `knn.py`
Implements K-Nearest Neighbors classification for healthcare record linkage.

### `knn_upsample.py`
Evaluates KNN using upsampled minority class data.

### `knn_downsample.py`
Evaluates KNN using downsampled majority class data.

---

# Single-layer Neural Network Files

### `snn.py`
Implements Single-layer Neural Network (SLNN) classification using the original dataset.

### `snn_upsample.py`
Evaluates SLNN performance using upsampled training data.

### `snn_downsample.py`
Evaluates SLNN performance using downsampled training data.

---

# Multi-layer Neural Network Files

### `mlnn.py`
Implements Multi-layer Neural Network (MLNN) classification using the original dataset.

### `mlnn_upsample.py`
Evaluates MLNN using upsampled training data.

### `mlnn_downsample.py`
Evaluates MLNN using downsampled training data.

---

# XGBoost Files

### `xgboost.py`
Implements XGBoost classification for healthcare record linkage tasks.

### `xgboosttuned.py`
Performs tuned XGBoost experiments with optimized parameters.

### `XGB_upsample.py`
Evaluates XGBoost using upsampled datasets.

### `XGB_downsample.py`
Evaluates XGBoost using downsampled datasets.

---

# Statistical Evaluation Files

### `Statistical_tests.py`
Performs statistical significance testing across machine learning model results.

### `mcnemar_sampling.py`
Performs McNemar testing to compare sampling strategy performance across models.

### `pprl_recall.py`
Analyzes recall-focused evaluation and threshold experimentation.

### `PPRL_train.py`
Handles training workflows, evaluation orchestration, and experimental execution.

---

# Research Context

This project was completed during a practicum internship at:

## Regenstrief Institute
Indianapolis, Indiana

in collaboration with:

## Indiana University
Luddy School of Informatics, Computing, and Engineering

Focus areas included:
- Healthcare AI
- Privacy-preserving analytics
- Healthcare data integration
- Machine learning for healthcare informatics
- Patient identity resolution

---

# Future Work

Future directions include:
- Evaluation using larger healthcare datasets
- Deep learning-based linkage architectures
- Hybrid probabilistic + machine learning linkage systems
- Adaptive threshold optimization
- Scalable healthcare deployment frameworks
- Human-in-the-loop linkage validation systems

---

# Author

## Lalitha Srihitha Prayaga

M.S. Health Informatics  
Indiana University Indianapolis

Practicum Internship:
Regenstrief Institute

---

# License

This project is licensed under the MIT License.

---
