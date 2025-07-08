# hiPCA: Health Index with Principal Component Analysis

## Introduction

hiPCA is a statistical framework for personalized health monitoring using gut microbiome data. It transforms high-dimensional microbiome compositions into interpretable health indexes that can:

- Differentiate between healthy and unhealthy states
- Identify microbial patterns associated with health
- Provide personalized health assessments
- Discover potential microbiome-based biomarkers

## Scripts Overview

### 1. `train_hipca.py`

**Purpose**: Train a new hiPCA model using microbiome data

**Usage**:
```bash
python train_hipca.py \
  --model_name my_model \
  --input microbiome_data.tsv \
  --metadata sample_metadata.tsv \
  --sample_col sample_id \
  --diagnosis_col health_status \
  --control_label healthy
```

**Inputs**:
- Microbiome abundance data (TSV format, samples as columns, features as rows)
- Sample metadata with health status labels
- Control group label (e.g., "healthy")

**Outputs**:
- Saved model components in `model_data/my_model/`:
  - PCA model
  - Scaling parameters
  - Threshold values
  - Feature matrices

### 2. `evaluate_hipca.py`

**Purpose**: Evaluate new samples using a pre-trained hiPCA model

**Usage**:
```bash
python evaluate_hipca.py \
  --path model_data/my_model \
  --input new_samples.tsv \
  --outdir results
```

**Inputs**:
- Pre-trained model directory
- New microbiome data (same format as training data)

**Outputs**:
- `hiPCA_results.csv` containing:
  - T², Q, and Combined index values
  - Health predictions for each sample

### 3. `hipca_utils.py`

**Purpose**: Core hiPCA functionality (used by other scripts)

**Contains**:
- Data transformation functions
- PCA modeling
- Index calculations
- Statistical methods

## Example Workflow

1. **Train a model**:
```bash
python train_hipca.py --model_name gut_health --input gut_data.tsv \
  --metadata metadata.csv --sample_col sample --diagnosis_col status --control_label healthy
```

2. **Evaluate new samples**:
```bash
python evaluate_hipca.py --path model_data/gut_health --input new_samples.tsv --outdir patient_results
```

## Requirements

- Python 3.7+
- Required packages:
  - pandas
  - numpy
  - scipy
  - scikit-learn
  - joblib

Install requirements with:
```bash
pip install pandas numpy scipy scikit-learn joblib
```

## Model Interpretation

The hiPCA model produces three key indexes:

1. **Hotelling's T²**: Measures variation in principal component space
2. **Q statistic**: Captures residual variation
3. **Combined index**: Weighted combination of T² and Q

Samples are classified as:
- **Healthy**: All indexes below thresholds
- **Unhealthy**: Any index above threshold
