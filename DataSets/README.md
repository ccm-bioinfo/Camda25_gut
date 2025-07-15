# CAMDA Gut Microbiome Health Index Challenge Repository

## Overview

This repository contains the complete analysis pipeline and results for the CAMDA (Critical Assessment of Massive Data Analysis) Gut Microbiome Health Index Challenge, spanning both 2024 and 2025 competitions. The challenge focuses on developing novel approaches to assess microbiome health from stool samples, with emphasis on the **Theatre of Activity (ToA)** concept that considers not just microbiota but the entire ecosystem of microbiome functions and interactions.

## Repository Structure

```
├── DataSets/                    # Raw and processed datasets
├── images/                      # Visualization outputs and figures
├── indexes/                     # Health index models and implementations
│   ├── dashboard.py             # 🌐 Interactive web calculator (streamlit run dashboard.py)
│   ├── ENSEMBLE/
│   ├── GMHI/
│   └── hiPCA/
├── output/                      # Analysis results and computed indices
├── scripts/                     # Analysis scripts and notebooks
└── README.md                   # This file
```

## 📊 DataSets

### CAMDA_2024 (613 samples)
- **BrackenAnnotation/**: Taxonomic profiles using Bracken
- **COVID/**: COVID-19 vs healthy classification dataset (35 patients)
- **MetacycAnnotation/**: Functional pathway annotations
  - `cummulative/`: Aggregated pathway data (scaled/unscaled)
  - `noncummulative/`: Individual pathway data (scaled/unscaled)
- **MifaserAnnotation/**: Alternative functional annotation using Mifaser

### CAMDA_2025 (4,398 samples)
- **networks/**: Network analysis data and adjacency matrices
- **taxa_category/**: Taxonomic categorization and metadata

### q2-dysbiosis_test
- Additional test dataset for dysbiosis analysis

## 🔬 Indexes

Implementation and models for various microbiome health indices:

### ENSEMBLE
- Ensemble learning approaches combining multiple indices
- `models/`: Trained ensemble models
- `output/`: Ensemble predictions

### GMHI (Gut Microbiome Health Index)
- Original GMHI implementation and analysis
- `model_data/`: Training data and bacterial species lists
- `notebooks/`: Analysis notebooks
- `output/`: GMHI scores and classifications

### hiPCA
- hiPCA index implementation with extensive model variations
- `model_data/`: Contains multiple dataset variations:
  - `zhu_model/`: Original Zhu et al. model implementation
  - `camda_all_samples/`: Full 2024 dataset
  - `ID_CAMDA_2025/`: Infectious disease-specific samples
  - `MBD_CAMDA_2025/`: Mental disorder-specific samples

## 📈 Output

### CAMDA_2024 Results
- **centrality_measures/**: Network centrality analysis
- **differential_otus/**: Differentially abundant taxa
- **GMHI/** & **hiPCA/**: Index calculations and scores
- **INDEX/**: Combined index results
  - `GMHI/`: GMHI-specific outputs
  - `hiPCA/`: hiPCA-specific outputs
- **Keystone/**: Keystone species analysis
  - `connected_otus/`: Network connectivity analysis
- **MD_index/**: MD index calculation scripts

### CAMDA_2025 Results
- **densidad_pathways/**: Pathway density analysis
- **diversity_plots/**: Alpha and beta diversity visualizations
- **GMHI/** & **hiPCA/**: List of species considered on each model
- **networks/**: Network analysis outputs
  - `centrality_measures/`: Node centrality metrics
  - `generate_network/`: Network generation results
  - `key.taxon_network_O_spcor_CAMDA25/`: Key taxon network analysis
- **spieceasi_network/**: SPIEC-EASI network inference results

## 🔍 Key Bacterial Species and Taxa

### 🎯 **IMPORTANT: Bacterial Species Used in Index Calculations**

For researchers interested in the specific bacterial species used in health index calculations:


#### Results and Bacterial Contributions
- **CAMDA_2025 hiPCA Results**: `output/CAMDA_2025/hiPCA/`
- **CAMDA_2025 GMHI Results**: `output/CAMDA_2025/GMHI/`


## 🚀 Getting Started

### Prerequisites
- R (version 4.0+)
- Python (version 3.8+)
- Streamlit (for web dashboard)
- Required R packages: microbiome, phyloseq, vegan, igraph
- Required Python packages: pandas, numpy, scikit-learn, networkx, streamlit


### 🌐 **Interactive Web Dashboard**

#### **Gut Microbiome Health Calculator**
Run the interactive web application to easily calculate health indices using the developed models:

```bash
cd indexes
streamlit run dashboard.py
```

**Features:**
- Interactive calculation of GMHI, hiPCA, and ensemble indices
- Upload your own microbiome data
- Easy-to-use interface for non-technical users
- Comparison between different health indices


## 📚 References

- Gupta, V. K., et al. (2020). GMHI development
- Zhu, Q., et al. (2023). hiPCA index methodology
- Berg, G., et al. (2020). Theatre of Activity concept
- Chang, Y., et al. (2024). GMWI2 advancement

## 🤝 Contributing

This repository contains the complete analysis pipeline for the CAMDA challenges. Researchers can:
- Explore the bacterial species used in index calculations
- Reproduce the analysis using provided scripts
- Use the **interactive web dashboard** (`cd indexes && streamlit run dashboard.py`) for easy index calculations
- Extend the methodologies for new datasets
- Develop novel health indices based on the Theatre of Activity concept

## 🌐 Web Application Usage

The `indexes/dashboard.py` provides an intuitive interface for:
- **Researchers**: Quick validation of results and hypothesis testing
- **Clinicians**: Easy-to-use tool for microbiome health assessment
- **Students**: Educational tool for understanding microbiome indices
- **Collaborators**: Sharing results and demonstrating methodologies

## 📞 Contact

For questions about the datasets, methodologies, or bacterial species used in calculations, please refer to the specific model data directories or contact the challenge organizers.