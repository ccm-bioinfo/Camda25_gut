"""
hiPCA (Health Index with Principal Component Analysis)
A statistical framework for personalized health monitoring using gut microbiome data
"""

import pandas as pd
import numpy as np
from scipy.stats import kstest, chi2, norm
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import IsolationForest
import argparse
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

def ks_test(df, healthy, non_healthy, method_ks='asymp', p_val=0.001):
    """
    Perform Kolmogorov-Smirnov test to identify health-associated microbial features
    
    Args:
        df: DataFrame containing microbial abundance data
        healthy: List of healthy sample IDs
        non_healthy: List of non-healthy sample IDs
        method_ks: KS test method ('asymp', 'exact', 'approx')
        p_val: Significance threshold
        
    Returns:
        Tuple of (healthy_features, nonhealthy_features)
    """
    healthy_df = df[[x for x in df.columns if x in healthy]].T
    nonhealthy_df = df[[x for x in df.columns if x in non_healthy]].T
    
    healthy_features = []
    nonhealthy_features = []
    
    for feature in df.index:
        # Test for features more abundant in healthy
        if kstest(healthy_df[feature], nonhealthy_df[feature], 
                 alternative='less', method=method_ks).pvalue <= p_val:
            healthy_features.append(feature)
            
        # Test for features more abundant in non-healthy
        if kstest(nonhealthy_df[feature], healthy_df[feature],
                 alternative='less', method=method_ks).pvalue <= p_val:
            nonhealthy_features.append(feature)
            
    print(f'# Healthy features selected by KS: {len(healthy_features)}')
    print(f'# Unhealthy features selected by KS: {len(nonhealthy_features)}')
    return healthy_features, nonhealthy_features

def custom_transform(x):
    """Custom transformation for microbiome abundance data"""
    return np.log2(2 * x + 0.00001) if x <= 1 else np.sqrt(x)

def transform_data(df, features, scaler=None):
    """
    Transform and scale microbiome data
    
    Args:
        df: Raw abundance DataFrame
        features: List of features to include
        scaler: Optional pre-fit scaler
        
    Returns:
        Tuple of (transformed_data, scaling_parameters, scaler)
    """
    # Create dataframe with selected features
    aux = pd.DataFrame()
    for feature in set(features):
        aux[feature] = df.T[feature] if feature in df.index else [0]*len(df.T)
    
    # Apply custom transformation
    transformed = aux.applymap(custom_transform)
    
    # Standardize data
    if scaler is None:
        scaler = StandardScaler()
        scaler.fit(transformed)
    
    scaled_data = scaler.transform(transformed)
    scaled_df = pd.DataFrame(scaled_data, columns=transformed.columns)
    scaled_df.index = df.T.index
    
    # Store scaling parameters
    scaling_params = pd.DataFrame({
        'specie': transformed.columns,
        'mean': scaler.mean_,
        'std': scaler.scale_
    })
    
    return scaled_df, scaling_params, scaler

def calculate_pca_stats(df, variance_for_pc=0.9, alpha=0.05):
    """
    Perform PCA and calculate monitoring statistics
    
    Args:
        df: Transformed microbiome data
        variance_for_pc: Variance threshold for PC selection
        alpha: Significance level for thresholds
        
    Returns:
        Tuple of PCA results and monitoring parameters
    """
    pca = PCA()
    pca.fit(df)
    
    # Extract PCA components
    eigenvalues = pca.explained_variance_
    eigenvectors = pca.components_
    
    # Create PCA summary dataframe
    pca_data = pd.DataFrame({
        'Eigenvectors': eigenvectors,
        'Explained_variance': eigenvalues,
        '%variance': eigenvalues / eigenvalues.sum()
    }).sort_values('%variance', ascending=False)
    
    pca_data['%variance_cumulative'] = pca_data['%variance'].cumsum()
    
    # Split into principal and residual components
    principal_mask = pca_data['%variance_cumulative'] < variance_for_pc
    principal_components = pca_data[principal_mask]['Eigenvectors']
    principal_values = pca_data[principal_mask]['Explained_variance']
    
    # Calculate Hotelling's T² parameters
    D = principal_components.T @ np.linalg.inv(np.diag(principal_values)) @ principal_components
    t2_threshold = chi2.ppf(1-alpha, len(principal_components))
    
    # Calculate Q statistic parameters
    residual_components = pca_data[~principal_mask]['Eigenvectors']
    C = residual_components.T @ residual_components
    Theta1 = pca_data[~principal_mask]['Explained_variance'].sum()
    Theta2 = (pca_data[~principal_mask]['Explained_variance']**2).sum()
    
    # Q threshold calculation
    h0 = 1 - (2*Theta1**3)/(3*Theta2**2)
    Q_alpha = (Theta2/Theta1) * chi2.ppf(1-alpha, len(residual_components)) * (Theta1**2/Theta2)
    
    # Combined index parameters
    fi = D/t2_threshold + C/Q_alpha
    g = ((len(principal_components)/t2_threshold**2) + (Theta2/Q_alpha**2)) / (
         (len(principal_components)/t2_threshold) + (Theta1/Q_alpha))
    h = ((len(principal_components)/t2_threshold) + (Theta1/Q_alpha))**2 / (
         (len(principal_components)/t2_threshold**2) + (Theta2/Q_alpha**2))
    threshold_combined = g * chi2.ppf(1-alpha, h)
    
    return pca, pca_data, D, t2_threshold, C, Q_alpha, fi, threshold_combined

def calculate_indexes(data, pca, D, C, fi, thresholds):
    """
    Calculate health monitoring indexes for samples
    
    Args:
        data: Transformed microbiome data
        pca: Fitted PCA model
        D: Hotelling's T² matrix
        C: Q statistic matrix  
        fi: Combined index matrix
        thresholds: Dictionary of threshold values
        
    Returns:
        DataFrame with index values and predictions
    """
    try:
        transformed = pca.transform(data)
    except:
        transformed = np.array(data)
    
    results = []
    for item in transformed:
        # Calculate all three indexes
        t2 = item.T @ D @ item
        q = item.T @ C @ item
        combined = item.T @ fi @ item
        
        # Make predictions
        pred_t2 = 'Healthy' if t2 <= thresholds['t2'] else 'Unhealthy'
        pred_q = 'Healthy' if q <= thresholds['q'] else 'Unhealthy'
        pred_combined = 'Healthy' if combined <= thresholds['combined'] else 'Unhealthy'
        
        results.append({
            'T2': t2,
            'Prediction T2': pred_t2,
            'Q': q, 
            'Prediction Q': pred_q,
            'Combined Index': combined,
            'Combined Prediction': pred_combined
        })
    
    return pd.DataFrame(results, index=data.index)

def hiPCA(df, healthy, non_healthy, features=None, ks=False, method='asymp', 
          p_val=0.001, only_nonhealthy=False):
    """
    Main hiPCA workflow
    
    Args:
        df: Raw microbiome data
        healthy: Healthy sample IDs
        non_healthy: Non-healthy sample IDs
        features: Optional pre-selected features
        ks: Whether to perform KS test feature selection
        method: KS test method
        p_val: KS test p-value threshold
        only_nonhealthy: Use only non-healthy associated features
        
    Returns:
        Tuple of hiPCA model components
    """
    # Feature selection
    if ks:
        healthy_features, nonhealthy_features = ks_test(
            df, healthy, non_healthy, method_ks=method, p_val=p_val)
        features = nonhealthy_features if only_nonhealthy else healthy_features + nonhealthy_features
    
    # Data transformation
    transformed, scaling_data, scaler = transform_data(
        df[[x for x in healthy if x in df.columns]], features)
    
    # PCA modeling
    pca, pca_data, D, t2_thresh, C, Q_alpha, fi, comb_thresh = calculate_pca_stats(transformed)
    
    print(f'T² threshold: {t2_thresh:.2f}, Q threshold: {Q_alpha:.2f}, '
          f'Combined threshold: {comb_thresh:.2f}')
    
    thresholds = {
        't2': t2_thresh,
        'q': Q_alpha, 
        'combined': comb_thresh
    }
    
    return (features, pca, pca_data, D, C, fi, thresholds, 
            transformed, scaling_data, scaler)

def main():
    """Main execution workflow"""
    parser = argparse.ArgumentParser(description="Train hiPCA model")
    parser.add_argument('--input', required=True, help="Path to input data")
    parser.add_argument('--metadata', required=True, help="Path to metadata")
    parser.add_argument('--sample', required=True, help="Sample ID column name")
    parser.add_argument('--diagnosis', required=True, help="Diagnosis column name")
    parser.add_argument('--control', required=True, help="Control sample identifier")
    parser.add_argument('--method', default='asymp', 
                       choices=['exact', 'approx', 'asymp'],
                       help='KS test computation method')
    
    args = parser.parse_args()
    
    # Load and prepare data
    data = pd.read_csv(args.input, sep='\t', index_col=0)
    metadata = pd.read_csv(args.metadata, sep='\t')
    metadata = metadata[metadata[args.sample].isin(data.columns)]
    
    # Define disease categories
    conditions = {
        'ID': ['acute_diarrhoea', 'STH', 'respiratoryinf'],
        'NP': ['adenoma', 'CRC', 'few_polyps', 'recipient_before'],
        'MD': ['hypercholesterolemia', 'IGT', 'T2D'],
        'MBD': ['schizophrenia', 'ME/CFS', 'PD'],
        'CD': ['ACVD', 'CAD', 'hypertension', 'BD'],
        'DD': ['CD', 'cirrhosis', 'IBD', 'UC'],
        'ALL': [x for x in metadata[args.diagnosis].unique() if x != args.control]
    }
    
    # Process each condition
    for condition, diseases in conditions.items():
        print(f"\nProcessing condition: {condition}")
        
        # Prepare data subsets
        metadata_sub = metadata[metadata['category'].isin(diseases)]
        healthy_samples = metadata[metadata[args.diagnosis] == args.control]
        
        # Balance healthy/non-healthy ratio
        iso = IsolationForest(random_state=42, contamination='auto')
        iso.fit(data[healthy_samples['sample']].T)
        scores = iso.decision_function(data[healthy_samples['sample']].T)
        healthy_selected = healthy_samples.iloc[np.argsort(scores)[:2*len(metadata_sub)]]
        
        # Create final dataset
        final_samples = list(metadata_sub['sample']) + list(healthy_selected['sample'])
        metadata_final = metadata[metadata['sample'].isin(final_samples)]
        data_final = data[final_samples]
        
        # Cross-validation setup
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        metadata_final['fold'] = -1
        for fold, (_, val_idx) in enumerate(skf.split(metadata_final, metadata_final[args.diagnosis])):
            metadata_final.loc[val_idx, 'fold'] = fold
        
        # Cross-validation loop
        results = []
        for fold in metadata_final['fold'].unique():
            print(f"  Fold {fold+1}/5")
            
            # Split data
            train_idx = metadata_final['fold'] != fold
            train_samples = metadata_final[train_idx][args.sample]
            test_samples = metadata_final[~train_idx][args.sample]
            
            train_data = data_final[train_samples]
            test_data = data_final[test_samples]
            
            # Get healthy/non-healthy samples
            healthy = metadata_final[train_idx & 
                    (metadata_final[args.diagnosis] == args.control)][args.sample]
            non_healthy = metadata_final[train_idx & 
                       (metadata_final[args.diagnosis] != args.control)][args.sample]
            
            # Train hiPCA model
            model_output = hiPCA(train_data, healthy, non_healthy, 
                               ks=True, method=args.method, 
                               only_nonhealthy=True)
            
            # Evaluate on test set
            test_transformed = transform_data(test_data, model_output[7], model_output[9])[0]
            test_results = calculate_indexes(test_transformed, *model_output[1:7])
            
            # Calculate performance
            true_labels = (metadata_final[~train_idx][args.diagnosis] == args.control).astype(int)
            preds = {
                'T2': [1 if x == 'Healthy' else 0 for x in test_results['Prediction T2']],
                'Q': [1 if x == 'Healthy' else 0 for x in test_results['Prediction Q']],
                'Combined': [1 if x == 'Healthy' else 0 for x in test_results['Combined Prediction']]
            }
            
            fold_results = {
                'Fold': fold+1,
                'T2_F1': f1_score(true_labels, preds['T2'], pos_label=0),
                'Q_F1': f1_score(true_labels, preds['Q'], pos_label=0),
                'Combined_F1': f1_score(true_labels, preds['Combined'], pos_label=0)
            }
            results.append(fold_results)
        
        # Save results
        results_df = pd.DataFrame(results)
        results_df.to_csv(f'output/{condition}_results.csv', index=False)
        print(f"Saved results for {condition}")

if __name__ == "__main__":
    main()