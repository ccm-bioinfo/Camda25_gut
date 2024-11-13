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


## Repository Contents

```
├── Camda25_gut
│   ├── DataSets
│   │   ├── CAMDA
│   │   ├── COVID
│   │   └── INDEX 
│   ├── indexes
│   │   ├── hiPCA
│   │   └── RFT_GMHI
│   ├── output
│   │   ├── hiPCA
│   │   ├── GMHI
│   │   └── MD_index
│   └── scripts
│       └── ...
└── ...
```

## Run dasboard

```bash
git clone https://github.com/ccm-bioinfo/Camda25_gut.git
cd Camda25_gut/indexes/
streamlit run dashboard.py
```
