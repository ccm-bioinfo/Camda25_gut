# hiPCA: Health Index with Principal Component Analysis

## Introduction

hiPCA is a statistical framework for personalized health monitoring using gut microbiome data. It transforms high-dimensional microbiome compositions into interpretable health indexes that can:

- Differentiate between healthy and unhealthy states
- Identify microbial patterns associated with health
- Provide personalized health assessments
- Discover potential microbiome-based biomarkers

## Scripts Overview

### 1. `hiPCA_train.py`

**Purpose**: Train a new hiPCA model using microbiome data

**Usage**:
```bash
python hiPCA_train.py \
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

### 2. `hiPCA_calculate.py`

**Purpose**: Evaluate new samples using a pre-trained hiPCA model

**Usage**:
```bash
python hiPCA_calculate.py \
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

### 3. `hiPCA_evaluate_kfolds.py`

**Purpose**: Perform 5-fold stratified cross-validation evaluation of hiPCA model

**Usage**:
```bash
python hiPCA_evaluate_kfolds.py \
  --input microbiome_data.tsv \
  --metadata sample_metadata.tsv \
  --sample_col sample_id \
  --diagnosis_col health_status \
  --control_label healthy \
  --outdir kfold_results
```

**Inputs**:
- Same inputs as training script
- Uses stratified 5-fold cross-validation to ensure balanced splits

**Outputs**:
- Cross-validation performance metrics
- Per-fold results and statistics
- Overall model performance assessment



## Model Data Structure

The `model_data/` folder contains saved model components for each trained hiPCA model:

```
model_data/
└── model_name/
    ├── pca_model.pkl          # Trained PCA model
    ├── scaler.pkl             # Data scaling parameters
    ├── thresholds.json         # T² and Q threshold values
    ├── feature_matrices.npy     # Processed feature data
    └── scaling_parameters.pkl    # Data to scale the features

```

**Key Files**:
- **pca_model.pkl**: Scikit-learn PCA model with fitted parameters
- **scaler.pkl**: StandardScaler or other normalization parameters
- **thresholds.pkl**: Statistical thresholds for T² and Q indexes
- **feature_matrix.pkl**: Transformed feature data used for training
- **control_indices.pkl**: Indices of control samples used for threshold calculation
- **metadata.pkl**: Training parameters and model configuration

These files are automatically created during training and loaded during evaluation to ensure consistent data processing and model application.

## Example Workflow

1. **Train a model**:
```bash
python hiPCA_train.py --model_name gut_health --input gut_data.tsv \
  --metadata metadata.csv --sample_col sample --diagnosis_col status --control_label healthy
```

2. **Evaluate model performance with k-fold cross-validation**:
```bash
python hiPCA_evaluate_kfolds.py --input gut_data.tsv --metadata metadata.csv \
  --sample_col sample --diagnosis_col status --control_label healthy --outdir cv_results
```

3. **Calculate indexes for new samples**:
```bash
python hiPCA_calculate.py --path model_data/gut_health --input new_samples.tsv --outdir patient_results
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