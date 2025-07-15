# GMHI: Gut Microbiome Health Index Calculator

## Introduction

The Gut Microbiome Health Index (GMHI) is a quantitative metric for assessing gut microbiome health status based on the relative abundance of healthy and unhealthy marker species. This tool provides two analysis modes:

1. **Taxonomy-only analysis**: Uses only taxonomic abundance data
2. **Integrated analysis**: Combines taxonomic and pathway data with machine learning prediction

## Core Methodology

GMHI is calculated using the formula:
```
GMHI = log10(Ψ_MH / Ψ_MN)
```

Where:
- **Ψ_MH**: Psi value for healthy marker species
- **Ψ_MN**: Psi value for unhealthy marker species

The Psi value incorporates:
- Relative abundance of markers present in the sample
- Shannon entropy-like diversity measure
- Marker prevalence within the predefined marker sets

## Script Overview

### `gmhi_calculate.py`

**Purpose**: Calculate GMHI scores for microbiome samples using pre-trained marker species sets

## Usage

### Taxonomy-only Analysis

```bash
python gmhi_calculate.py \
  --path model_data/gmhi_model.csv \
  --taxonomy taxonomic_abundance.tsv \
  --outdir results
```

## Input Requirements

### Required Files

1. **Taxonomic abundance data** (`--taxonomy`):
   - Tab-separated file with species as rows and samples as columns
   - First column should contain species names
   - Values represent relative abundance

2. **Model directory** (`--path`):
   - Contains pre-trained marker species files and models


## Model Data Structure

The model directory should contain:

### For Taxonomy-only Analysis:
```
model_data/gmhi_model/
└── taxonomy.csv          # Healthy and unhealthy taxonomic markers
```

**Marker Files Format**:
Each CSV file should contain two columns:
- `Healthy`: List of healthy marker species
- `Unhealthy`: List of unhealthy marker species

## Outputs

### Taxonomy-only Analysis
- **File**: `GMHI_results.csv`
- **Contains**:
  - `GMHI`: Calculated GMHI score for each sample
  - `Prediction`: Health classification (Healthy if GMHI > 0, Unhealthy if GMHI ≤ 0)



## Requirements

- Python 3.7+
- Required packages:
  - pandas
  - numpy
  - pickle (standard library)
  - scikit-learn (for integrated analysis)

Install requirements:
```bash
pip install pandas numpy scikit-learn
```

## Interpretation

### GMHI Score Interpretation:
- **Positive GMHI (> 0)**: Indicates healthier gut microbiome
- **Negative GMHI (≤ 0)**: Indicates less healthy gut microbiome
- **Higher values**: Greater abundance of healthy markers relative to unhealthy markers

### Threshold Parameters:
- `theta_f = 1.4`: Global threshold parameter (reserved for future functionality)
- `theta_d = 0.1`: Global threshold parameter (reserved for future functionality)

## Technical Details

### Psi Calculation:
1. Find intersection of non-zero species in sample with marker set
2. Calculate relative marker presence: R_M = |markers_in_sample| / |total_markers|
3. Compute Shannon entropy-like term: Σ(n × ln(n))
4. Final Psi = (R_M / |total_markers|) × |Σ(n × ln(n))|
