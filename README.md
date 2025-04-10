# Gut Microbiome Health Index Challenge (CAMDA 2024)

## Overview

This challenge aimed to develop a novel gut microbiome-based health index that surpasses existing indices in performance. We leveraged data from the Human Microbiome Project 2 and two American Gut Project cohorts, incorporating the Theatre of Activity concept to enhance our models.

## Dataset Description

The dataset provided for the challenge includes 613 samples, sourced from:

- **Human Microbiome Project 2**
- **Two American Gut Project cohorts**

For each sample, the following data is provided:
- **Taxonomic Profiles:** Including health predictions from existing indices (Shannon Entropy, GMHI, and hiPCA).
- **Functional Profiles:** Detailing the microbial functions present in each sample.

Additionally we were provided with an additional dataset containing samples from 35 patients. The task is to explore the dataset and classify the patients into two groups: healthy controls and COVID-19 patients.

Each patient has two samples:
1. Sample from the day of admission.
2. The last sample from the ward stay.

The main objective of this challenge is to identify the healthy controls among the patients based on the provided dataset.


## Project Objectives

Our primary objectives for this challenge were to:
1. Develop a gut microbiome-based health index that outperforms existing indices.
2. Utilize the Theatre of Activity concept to improve health predictions.
3. Document our methods, results, and insights gained during the challenge.

## Requirements
- R (4.1.3)
- Python (3.10.5)

### Python libraries

To run this project, ensure you have the following Python libraries installed:

```bash
pandas
numpy
json
pickle
scipy
scikit-learn
joblib
```


## Repository Contents

```
├── Camda25_gut
│   ├── DataSets
│   │   ├── CAMDA
│   │   ├── COVID
│   │   └── INDEX 
│   ├── indexes
│   │   ├── hiPCA
│   │   └── RF_GMHI
│   ├── output
│   │   ├── hiPCA
│   │   ├── GMHI
│   │   └── MD_index
│   └── scripts
│       └── ...
└── ...
```

## Run dashboard

```bash
git clone https://github.com/ccm-bioinfo/Camda25_gut.git
cd Camda25_gut/indexes/
streamlit run dashboard.py
```

## Running the hiPCA Index

To calculate the hiPCA index, you can use either the CAMDA model or the original Zhu model. Here’s how to run the script:

   ```bash
   cd Camda25_gut/indexes/hiPCA
   python3 hiPCA_calculate.py --path model_data/{model}/ --input path/to/input --output output/directory/
   ```
## Running the GMHI Index

To calculate the RF-GMHI index, follow these steps:

```bash
cd Camda25_gut/indexes/RF-GMHI
python3 gmhi_calculate.py --path model_data/ --taxonomy path/to/input/taxonomy --pathways path/to/input/pathways
```
