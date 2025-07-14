import pandas as pd
import numpy as np
import pickle
import os
import argparse
from statistics import mean
from sklearn.ensemble import IsolationForest
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

import warnings
warnings.filterwarnings("ignore")

def parse_args():
    """
    Parse command line arguments for the hiPCA model training script.
    
    Returns:
        Parsed arguments object containing input paths and parameters
    """
    parser = argparse.ArgumentParser(description="Train hiPCA model")
    parser.add_argument('--input', required=True, type=str, help="Path to input data")
    parser.add_argument('--metadata', required=True, type=str, help="Path to metadata")
    parser.add_argument('--sample', required=True, type=str, help="Name of the column containing the ID of the sample in the metadata")
    parser.add_argument('--diagnosis', required=True, type=str, help="Name of the column containing the diagnosis in the metadata")
    parser.add_argument('--control', required=True, type=str, help="Identifier of the control sample")

    parser.add_argument(
        '--method',
        type=str,
        choices=['exact', 'approx', 'asymp'],
        default='asymp',
        help='Computation method for KS test to use'
    )

    return parser.parse_args()

def get_fH_fN_dH_dN(meta, tax):
    """
    Calculate prevalence metrics for each species in healthy vs non-healthy samples.
    
    Args:
        meta: DataFrame containing metadata with sample IDs and categories
        tax: DataFrame with species as rows and samples as columns (abundance data)
    
    Returns:
        metrics: DataFrame with f_H, f_N, d_H, d_N metrics for each species
                f_H = PH/PN (healthy prevalence ratio)
                f_N = PN/PH (non-healthy prevalence ratio)  
                d_H = PH-PN (healthy prevalence difference)
                d_N = PN-PH (non-healthy prevalence difference)
    """
    # Get sample IDs for healthy samples and extract their taxonomic data
    healthy_id = meta[meta['category'] == 'healthy']['sample']
    tax_healthy = tax[healthy_id]
    
    # Get sample IDs for non-healthy samples and extract their taxonomic data
    no_healthy_id = meta[meta['category'] != 'healthy']['sample']
    tax_no_healthy = tax[no_healthy_id]
    
    # Get all species from the taxonomic data
    species = tax.index
    
    # Define lower bound to avoid division by zero
    lower = 1e-05
    
    # Create DataFrame to store metrics with species as index
    metrics = pd.DataFrame(index=species, columns=['f_H', 'f_N', 'd_H', 'd_N'])
    
    # Calculate metrics for each species
    for specie in species:
        # Get species abundance in healthy samples and count non-zero occurrences
        specie_in_H = tax_healthy.loc[specie, :]
        abs_pres_H = len(specie_in_H[specie_in_H != 0])
        
        # Get species abundance in non-healthy samples and count non-zero occurrences
        specie_in_N = tax_no_healthy.loc[specie, :]
        abs_pres_N = len(specie_in_N[specie_in_N != 0])
        
        # Calculate prevalence rates (PH, PN) with lower bound protection
        PH = np.divide(abs_pres_H, len(specie_in_H), out=np.asanyarray(lower), where=(abs_pres_H != 0))
        PN = np.divide(abs_pres_N, len(specie_in_N), out=np.asanyarray(lower), where=(abs_pres_N != 0))
        
        # Calculate final metrics: ratios and differences
        metrics.loc[specie, :] = [np.divide(PH, PN), np.divide(PN, PH), PH - PN, PN - PH]
    
    return metrics

def get_MH_MN(metrics, theta_f, theta_d):
    """
    Identify healthy and unhealthy marker species based on threshold parameters.
    
    Args:
        metrics: DataFrame with f_H, f_N, d_H, d_N metrics for each species
        theta_f: Threshold for prevalence ratio (f_H or f_N)
        theta_d: Threshold for prevalence difference (d_H or d_N)
    
    Returns:
        MH: Set of healthy marker species (high in healthy samples)
        MN: Set of unhealthy marker species (high in non-healthy samples)
    """
    # Get healthy species that exceed both thresholds
    health_species_pass_theta_f = set(metrics[metrics['f_H'] >= theta_f].index)
    health_species_pass_theta_d = set(metrics[metrics['d_H'] >= theta_d].index)
    
    # Get unhealthy species that exceed both thresholds
    no_health_species_pass_theta_f = set(metrics[metrics['f_N'] >= theta_f].index)
    no_health_species_pass_theta_d = set(metrics[metrics['d_N'] >= theta_d].index)
    
    # Final marker sets: species that pass both thresholds
    MH = health_species_pass_theta_f & health_species_pass_theta_d
    MN = no_health_species_pass_theta_f & no_health_species_pass_theta_d
    
    return MH, MN

def get_Psi(set_M, sample):
    """
    Calculate Psi value for a given marker set and sample.
    
    Args:
        set_M: Set of marker species (either MH or MN)
        sample: Pandas Series containing species abundance data for a sample
    
    Returns:
        Psi: Calculated Psi value combining relative marker presence and Shannon entropy
    """
    # Find intersection of markers present in sample and marker set
    M_in_sample = set(sample[sample != 0].index) & set_M
    
    # Calculate relative abundance of markers in sample vs total markers in set
    R_M_sample = np.divide(len(M_in_sample), len(set_M))
    
    # Get abundance values for markers present in sample
    n = sample[sample != 0][list(M_in_sample)]
    log_n = np.log(n)
    # Calculate sum of n*ln(n) for Shannon entropy-like term
    sum_nlnn = np.sum(n * log_n)
    
    # Calculate final Psi value
    Psi = np.divide(R_M_sample, len(set_M)) * np.absolute(sum_nlnn)
    
    # Avoid division by zero in GMHI calculation
    if Psi == 0:
        Psi = 1e-05
    return Psi

def get_all_GMHI(tax, MH, MN):
    """
    Calculate Gut Microbiome Health Index (GMHI) for all samples.
    
    Args:
        tax: DataFrame with species as rows and samples as columns (abundance data)
        MH: Set of healthy marker species
        MN: Set of unhealthy marker species
    
    Returns:
        GMHI: Pandas Series with GMHI values for each sample
    """
    # Create series to store GMHI values for each sample
    samples = tax.columns 
    GMHI = pd.Series(index=samples, name='GMHI', dtype='float64')
    
    for sample in samples:
        # Calculate Psi for healthy and unhealthy markers
        Psi_MH = get_Psi(MH, tax[sample])
        Psi_MN = get_Psi(MN, tax[sample])
        
        # GMHI is log10 ratio of healthy to unhealthy Psi values
        GMHI_sample = np.log10(np.divide(Psi_MH, Psi_MN))
        GMHI[sample] = GMHI_sample
        
    return GMHI 

def get_accuracy2(GMHI, meta):
    """
    Calculate F1 score for GMHI predictions.
    
    Args:
        GMHI: Series of GMHI values
        meta: List of actual categories/diagnoses
    
    Returns:
        F1 score with unhealthy samples as positive label
    """
    # Convert categories to binary (0=unhealthy, 1=healthy)
    # Convert GMHI to binary predictions (0=unhealthy if <0, 1=healthy if >=0)
    return f1_score([0 if x != 'healthy' else 1 for x in meta], 
                    [0 if x < 0 else 1 for x in list(GMHI)], 
                    pos_label=0)

def main():
    """
    Main function to train and evaluate hiPCA model using cross-validation.
    Processes multiple disease conditions and generates GMHI predictions.
    """
    args = parse_args()
    
    # Load input data and metadata
    data = pd.read_csv(f'{args.input}', sep='\t', index_col=0)
    metadata = pd.read_csv(f'{args.metadata}', sep='\t')

    # Filter metadata to only include samples present in the data
    metadata = metadata[metadata[args.sample].isin(data.columns)]

    # Define disease condition groups for analysis
    conditions = dict()

    # Infectious diseases
    conditions['ID'] = [
        'acute_diarrhoea',
        'STH',
        'respiratoryinf'
    ]

    # Neoplastic conditions
    conditions['NP'] = [
        'adenoma',
        'adenoma;hypercholesterolemia',
        'adenoma;hypercholesterolemia;metastases',
        'adenoma;hypertension',
        'adenoma;metastases',
        'CRC',
        'CRC;hypercholesterolemia',
        'CRC;hypercholesterolemia;hypertension',
        'CRC;hypertension',
        'CRC;metastases',
        'CRC;T2D',
        'few_polyps',
        'T2D;adenoma',
        'recipient_before'
    ]

    # Metabolic disorders
    conditions['MD'] = [
        'hypercholesterolemia',
        'IGT',
        'IGT;MS',
        'metabolic_syndrome',
        'T2D',
        'T2D;respiratoryinf'
    ]

    # Mental/behavioral disorders
    conditions['MBD'] = [
        'schizophrenia',
        'ME/CFS',
        'PD'
    ]

    # Cardiovascular diseases
    conditions['CD'] = [
        'ACVD',
        'CAD',
        'CAD;T2D',
        'HF;CAD',
        'HF;CAD;T2D',
        'hypertension',
        'BD'
    ]

    # Digestive disorders
    conditions['DD'] = [
        'CD',
        'cirrhosis',
        'IBD',
        'UC'
    ]

    # All conditions combined
    conditions['ALL'] = [x for x in metadata[args.diagnosis].unique().tolist() if x != args.control]

    # Set threshold parameters for marker species selection
    theta_f = 1.4  # Prevalence ratio threshold
    theta_d = 0.0001  # Prevalence difference threshold (relaxed to avoid disadvantaging low-abundance species)

    # Prepare healthy control samples for balanced sampling
    metadata_healthy = metadata[metadata['category'] == 'healthy']
    data_healthy = data[list(metadata_healthy['sample'])]
    
    # Use Isolation Forest to identify most informative healthy samples
    iso = IsolationForest(random_state=42, contamination='auto')
    iso.fit(data_healthy.T)
    # Get anomaly scores (lower = more anomalous → more informative)
    scores = iso.decision_function(data_healthy.T)
    X_majority_df = data_healthy.T.copy()
    X_majority_df['score'] = scores
    # Sort by score (lowest = more anomalous → more informative)
    X_majority_sorted = X_majority_df.sort_values(by='score')

    # Process each disease condition
    for condition in conditions:
        # Get samples for current condition
        metadata_aux2 = metadata[metadata['category'].isin(conditions[condition])]
        n_minority = len(metadata_aux2)
        
        # Determine target number of healthy samples for 2:1 ratio
        n_majority_target = 2 * n_minority if 2 * n_minority < len(metadata_healthy) else len(metadata_healthy)
        
        # Select most informative healthy samples
        X_majority_selected = X_majority_sorted.iloc[:n_majority_target].drop(columns='score')
        healthy_indexes = X_majority_selected.index
        
        # Create balanced dataset with selected healthy samples and condition samples
        metadata_aux = metadata[metadata['sample'].isin(list(metadata_aux2['sample']) + list(healthy_indexes))]
        metadata_aux.reset_index(drop=True, inplace=True)   
        data_aux = data[list(metadata_aux['sample'])]

        # Set up 5-fold stratified cross-validation
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        metadata_aux['fold'] = -1
        
        # Assign fold numbers to samples
        for fold, (_, val_idx) in enumerate(skf.split(metadata_aux, metadata_aux[args.diagnosis])):
            metadata_aux.loc[val_idx, 'fold'] = fold
        
        # Initialize evaluation storage
        evaluations = []
        indexes = []
        
        print('---------------------------------------')
        print(condition)
        
        # Process each fold
        for fold in metadata_aux['fold'].unique():
            print(fold)

            # Split data into training and evaluation sets
            metadata_train = metadata_aux[metadata_aux['fold'] != fold]
            samples_train = list(metadata_train[args.sample])
            metadata_evaluate = metadata_aux[metadata_aux['fold'] == fold]
            samples_evaluate = list(metadata_evaluate[args.sample])

            data_train = data_aux[samples_train]
            data_evaluate = data_aux[samples_evaluate]

            # Calculate species metrics on training data
            metrics = get_fH_fN_dH_dN(metadata_train, data_train)

            # Identify marker species
            MH, MN = get_MH_MN(metrics, theta_f, theta_d)
            
            # Save marker species for this fold
            max_len = max(len(MH), len(MN))
            MH_padded = list(MH) + [None] * (max_len - len(MH))
            MN_padded = list(MN) + [None] * (max_len - len(MN))

            bacterias = pd.DataFrame({
                'Healthy': MH_padded,
                'Unhealthy': MN_padded
            })
            bacterias.to_csv(f'output/{condition}_bacterias_{fold}.csv')
            
            # Calculate GMHI for evaluation samples
            GMHI = get_all_GMHI(data_evaluate, MH, MN)
            indexes.append(GMHI)
            
            # Get true labels for evaluation samples
            y_test = []
            for sample in samples_evaluate:
                y_test.append(metadata[metadata['sample'] == sample]['category'].iloc[0])
            
            # Calculate accuracy
            accuracy = get_accuracy2(GMHI, y_test)
            print(accuracy)
            evaluations.append(accuracy)

        # Report results for this condition
        print(evaluations)
        print(f'Mean accuracy for {condition}: {mean(evaluations)}')
        
        # Save combined GMHI predictions from all folds
        GMHI_tax = pd.concat(indexes)
        GMHI_tax.to_csv(f'output/{condition}_GMHI_camda2025_preds_taxonomy_corrected_balanced.csv')

if __name__ == "__main__":
    main()