# CAMDA Gut Microbiome Health Index Challenge

## Overview

This repository contains datasets for the CAMDA (Critical Assessment of Massive Data Analysis) Gut Microbiome Health Index Challenge, spanning two years of research focused on developing novel approaches to assess microbiome health from stool samples.

The challenge addresses the growing need for non-invasive diagnostic tools as diseases linked to microbiome health, such as obesity and Inflammatory Bowel Disease (IBD), continue to rise. Stool samples offer a promising alternative to traditional diagnostic methods as they can be collected non-invasively, frequently, and are becoming increasingly affordable.

## Dataset Structure

```
├── CAMDA_2024/          # 2024 Challenge Dataset
├── CAMDA_2025/          # 2025 Challenge Dataset  
├── q2-dysbiosis_test/   # Additional test data
└── README.md           # This file
```

## CAMDA_2024 Dataset

### Challenge Objective
Develop a gut microbiome-based health index that outperforms existing indices by leveraging the **Theatre of Activity (ToA)** concept, which emphasizes not just the microbiota but the entire ecosystem of microbiome functions and interactions.

### Dataset Details
- **Size**: 613 samples
- **Source**: 
  - Human Microbiome Project 2
  - Two American Gut Project cohorts
- **Included Data**:
  - Precomputed taxonomic profiles
  - Health predictions from existing indices:
    - Shannon entropy
    - Gut Microbiome Health Index (GMHI)
    - hiPCA
  - Functional profiles

### Additional COVID-19 Dataset (Added April 24, 2024)
- **Objective**: Classify 35 patients into healthy controls vs. COVID-19 patients
- **Sample Structure**: 
  - 2 samples per patient
  - Sample A: Day of admission
  - Sample B: Last sample from ward stay (or second sample for healthy controls)

## CAMDA_2025 Dataset

### Challenge Objective
Build upon the Theatre of Activity concept with emphasis on:
- Developing novel ways to **combine taxonomic and functional profiles**
- **Exploring synergies** between different microbiome components
- Creative perspectives that advance understanding of microbiome in health and disease

### Dataset Details
- **Size**: 4,398 samples
- **Source**: Numerous cohorts with various diseases from the curated MetagenomicsData database
- **Included Data**:
  - Precomputed taxonomic profiles
  - Health predictions from existing indices:
    - Shannon entropy (on species and functions)
    - Gut Microbiome Health Index (GMHI)
    - Gut Microbiome Wellness Index (GMWI2)
    - hiPCA
  - Functional profiles

### Key Focus Areas
1. **Integration**: Novel approaches to combining taxonomic and functional data
2. **Synergies**: Exploring interactions between different microbiome components
3. **Innovation**: Creative methodologies that advance microbiome health understanding

## Background

### Current Approaches
Traditional microbiome health assessment relies on:
- **Alpha diversity**: Closely related to dysbiosis
- **Microbiome richness**: Key component of microbiome health and robustness
- **Existing indices**: GMHI, and hiPCA based on beneficial/harmful bacteria ratios

### Theatre of Activity (ToA) Concept
The modern definition of microbiome extends beyond just microbiota to include:
- **Microbiome functions**
- **Microbiota interactions**
- **Complete ecosystem dynamics**

## References

- Gupta, V. K., et al. (2020). A predictive index for health status using species-level gut microbiome profiling.
- Chang, Y., et al. (2024). Gut Microbiome Wellness Index 2. Nature Communications.
- Zhu, Q., et al. (2023). hiPCA index development.
- Berg, G., et al. (2020). Microbiome definition revisited: old concepts and new challenges.

## Usage

Each dataset folder contains the necessary files for analysis including taxonomic profiles, functional profiles, and metadata. 

## Challenge Goals

- **Primary**: Develop novel gut microbiome health indices
- **Secondary**: Advance understanding of microbiome-health relationships
- **Innovation**: Creative integration of taxonomic and functional data