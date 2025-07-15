# Ensemble Model: hiPCA + GMHI + Pathways Stacking Classifier

## Introduction

This ensemble model combines predictions from multiple microbiome analysis approaches to improve classification accuracy for gut health assessment. The model uses a stacking ensemble approach that integrates:

1. **hiPCA (Health Index with Principal Component Analysis)** predictions
2. **GMHI (Gut Microbiome Health Index)** predictions  
3. **Grouped pathway abundance data**

A Logistic Regression meta-classifier is trained on these combined features to make final binary health predictions (healthy vs. unhealthy).

## Methodology

### Stacking Ensemble Architecture

```
Input Layer:
├── hiPCA predictions (T², Q, Combined indexes)
├── GMHI predictions (health scores)
└── Grouped pathways (abundance features)
                    ↓
Meta-Classifier: Logistic Regression with StandardScaler
                    ↓
Final Output: Binary health classification
```

### Cross-Validation Strategy

- **5-fold Stratified Cross-Validation**: Ensures balanced representation of classes across folds
- **Pipeline**: StandardScaler → LogisticRegression
- **Evaluation Metric**: F1-score (with unhealthy as positive class)

## Script Overview

### `ensemble_model.py`

**Purpose**: Train and evaluate stacking ensemble model combining hiPCA, GMHI, and pathway features

## Input Requirements

### Required Files

1. **hiPCA predictions** (`../hiPCA/output/{condition}_camda2025_preds_taxonomy_corrected_balanced_final.csv`):
   - Contains T², Q, and Combined index predictions
   - Individual hiPCA model predictions for each sample

2. **GMHI predictions** (`../RF_GMHI/output/{condition}_GMHI_camda2025_preds_taxonomy_corrected_balanced.csv`):
   - Contains GMHI health scores
   - Gut microbiome health index predictions

3. **Metadata** (`../../DataSets/CAMDA_2025/metadata_corrected_final.txt`):
   - Sample metadata with true health labels
   - Tab-separated format with 'sample' and 'category' columns

4. **Grouped pathways** (`../../DataSets/Path_agrupados.txt`):
   - Pathway abundance data aggregated by functional groups
   - Tab-separated format with pathways as rows, samples as columns

## Usage

```bash
python ensemble_model.py
```

**Note**: The script is currently configured with hardcoded paths. Modify the file paths in the script to match your data structure.

## Data Processing Pipeline

### 1. Data Loading and Integration
```python
# Load predictions from individual models
hipca_preds = pd.read_csv(hipca_path, index_col=0)
gmhi_preds = pd.read_csv(gmhi_path, index_col=0)
pathways = pd.read_csv(pathways_path, sep='\t', index_col=0)

# Join all features on sample index
X = hipca_preds.join(gmhi_preds, how='inner').join(pathways, how='inner')
```

### 2. Feature Preparation
- Removes prediction columns: `Group`, `Prediction T2`, `Prediction Q`, `Combined Prediction`
- Keeps numerical scores: hiPCA indexes, GMHI scores, pathway abundances
- Converts multi-class labels to binary (healthy vs. unhealthy)

### 3. Cross-Validation Setup
- 5-fold stratified split ensuring balanced class distribution
- Pipeline with StandardScaler and LogisticRegression
- Random state = 42 for reproducibility

## Outputs

### 1. Prediction Results
**File**: `output/{condition}_binary_ensemble_predictions_logreg_final.tsv`
- `true_label`: Ground truth labels
- `predicted_label`: Ensemble model predictions
- `proba_healthy`: Probability of healthy class

### 2. Performance Metrics
**File**: `output/{condition}_binary_ensemble_f1_scores_logreg_final.tsv`
- F1-scores for each fold
- Mean F1-score across all folds

### 3. Confusion Matrix
**File**: `output/{condition}_binary_confusion_matrix_logreg_final.tsv`
- Binary confusion matrix (Healthy=0, Unhealthy=1)
- True vs. predicted class counts

### 4. Feature Importance
**File**: `output/{condition}_logreg_feature_importances_by_fold.tsv`
- Logistic regression coefficients for each feature
- Feature importance across all folds
- Identifies most informative features for classification

## Model Architecture Details

### Meta-Classifier Pipeline
```python
Pipeline([
    ('scaler', StandardScaler()),        # Feature normalization
    ('logreg', LogisticRegression(       # Linear classifier
        solver='liblinear', 
        random_state=42
    ))
])
```

### Feature Types
1. **hiPCA Features**: 
   - T² index (Hotelling's T²)
   - Q statistic (residual variation)
   - Combined index (weighted combination)

2. **GMHI Features**:
   - GMHI score (log ratio of healthy/unhealthy markers)
   - Additional GMHI-derived metrics

3. **Pathway Features**:
   - Grouped functional pathway abundances
   - Metabolic pathway activities
   - Microbial functional capacity

## Performance Evaluation

### Metrics
- **F1-Score**: Harmonic mean of precision and recall (unhealthy as positive class)
- **Confusion Matrix**: Classification accuracy breakdown
- **Cross-Validation**: 5-fold stratified validation


## Requirements

- Python 3.7+
- Required packages:
  - pandas
  - numpy
  - scikit-learn
  - sklearn (included in scikit-learn)

Install requirements:
```bash
pip install pandas numpy scikit-learn
```

## Example Workflow

1. **Generate individual model predictions**:
   ```bash
   # Run hiPCA model
   python hiPCA_calculate.py --path model_data/hipca_model --input data.csv --outdir hipca_output
   
   # Run GMHI model
   python gmhi_calculate.py --path model_data/gmhi_model --taxonomy data.tsv --outdir gmhi_output
   ```

2. **Run ensemble model**:
   ```bash
   python ensemble_model.py
   ```


## Configuration Options

### Customizable Parameters
```python
# Cross-validation settings
n_splits = 5
random_state = 42

# Logistic regression parameters
solver = 'liblinear'
penalty = 'l2'  # Can be modified for regularization

# Condition variable
condition = 'ALL'  # Can be modified for different analysis conditions
```
