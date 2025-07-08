"""
hiPCA (Health Index with Principal Component Analysis)
A statistical framework for personalized health monitoring using gut microbiome data
"""

import pandas as pd
import numpy as np
from scipy.stats import kstest, chi2, norm
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import argparse
import os
import json
import pickle

def ks_test(df, healthy_samples, non_healthy_samples, method='asymp', p_val=0.001):
    """
    Perform Kolmogorov-Smirnov test to identify health-associated microbial features
    
    Args:
        df: DataFrame containing microbial abundance data
        healthy_samples: List of healthy sample IDs
        non_healthy_samples: List of non-healthy sample IDs
        method: KS test method ('asymp', 'exact', 'approx')
        p_val: Significance threshold
        
    Returns:
        Tuple of (healthy_features, nonhealthy_features)
    """
    healthy_df = df[healthy_samples].T
    nonhealthy_df = df[non_healthy_samples].T
    
    healthy_features = []
    nonhealthy_features = []
    
    for feature in df.index:
        # Test for features more abundant in healthy
        healthy_pval = kstest(healthy_df[feature], nonhealthy_df[feature], 
                             alternative='less', method=method).pvalue
        if healthy_pval <= p_val:
            healthy_features.append(feature)
            
        # Test for features more abundant in non-healthy
        nonhealthy_pval = kstest(nonhealthy_df[feature], healthy_df[feature],
                                alternative='less', method=method).pvalue
        if nonhealthy_pval <= p_val:
            nonhealthy_features.append(feature)
            
    print(f'Healthy features selected: {len(healthy_features)}')
    print(f'Unhealthy features selected: {len(nonhealthy_features)}')
    return healthy_features, nonhealthy_features

def custom_transform(x):
    """Custom transformation for microbiome abundance data"""
    return np.log2(2 * x + 1e-5) if x <= 1 else np.sqrt(x)

def transform_and_scale(df, features, model_dir):
    """
    Transform and scale microbiome data
    
    Args:
        df: Raw abundance DataFrame
        features: List of features to include
        model_dir: Directory to save scaling parameters
        
    Returns:
        Tuple of (transformed_data, scaler)
    """
    # Create dataframe with selected features
    data = pd.DataFrame()
    for feature in set(features):
        data[feature] = df.T[feature] if feature in df.index else 0
    
    # Apply custom transformation
    transformed = data.applymap(custom_transform)
    
    # Standardize data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(transformed)
    
    # Save scaling parameters
    os.makedirs(model_dir, exist_ok=True)
    with open(f'{model_dir}/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    scaling_params = pd.DataFrame({
        'specie': transformed.columns,
        'mean': scaler.mean_,
        'std': scaler.scale_
    })
    scaling_params.to_csv(f'{model_dir}/scaling_parameters.csv', index=False)
    
    return pd.DataFrame(scaled_data, columns=transformed.columns, index=df.T.index), scaler

def calculate_pca_model(data, variance_threshold=0.9, alpha=0.05, model_dir=None):
    """
    Perform PCA and calculate monitoring statistics
    
    Args:
        data: Transformed microbiome data
        variance_threshold: Variance threshold for PC selection
        alpha: Significance level for thresholds
        model_dir: Directory to save model outputs
        
    Returns:
        Tuple of PCA model and monitoring parameters
    """
    pca = PCA()
    pca.fit(data)
    
    if model_dir:
        with open(f'{model_dir}/pca_model.pkl', 'wb') as f:
            pickle.dump(pca, f)
    
    # Create PCA summary
    eigenvalues = pca.explained_variance_
    pca_data = pd.DataFrame({
        'Eigenvectors': list(pca.components_),
        'Explained_variance': eigenvalues,
        '%variance': eigenvalues / eigenvalues.sum()
    }).sort_values('%variance', ascending=False)
    pca_data['%variance_cumulative'] = pca_data['%variance'].cumsum()
    
    # Split components
    principal_mask = pca_data['%variance_cumulative'] < variance_threshold
    principal_components = pca_data[principal_mask]['Eigenvectors']
    principal_values = pca_data[principal_mask]['Explained_variance']
    
    # Hotelling's T² parameters
    D = principal_components.T @ np.linalg.inv(np.diag(principal_values)) @ principal_components
    t2_threshold = chi2.ppf(1-alpha, len(principal_components))
    
    # Q statistic parameters
    residual_components = pca_data[~principal_mask]['Eigenvectors']
    C = residual_components.T @ residual_components
    Theta1 = pca_data[~principal_mask]['Explained_variance'].sum()
    Theta2 = (pca_data[~principal_mask]['Explained_variance']**2).sum()
    
    # Q threshold calculation
    Q_alpha = (Theta2/Theta1) * chi2.ppf(1-alpha, len(residual_components)) * (Theta1**2/Theta2)
    
    # Combined index parameters
    fi = D/t2_threshold + C/Q_alpha
    g = ((len(principal_components)/t2_threshold**2) + (Theta2/Q_alpha**2)) / (
         (len(principal_components)/t2_threshold) + (Theta1/Q_alpha))
    h = ((len(principal_components)/t2_threshold) + (Theta1/Q_alpha))**2 / (
         (len(principal_components)/t2_threshold**2) + (Theta2/Q_alpha**2))
    threshold_combined = g * chi2.ppf(1-alpha, h)
    
    # Save model outputs
    if model_dir:
        np.save(f'{model_dir}/D_matrix.npy', D)
        np.save(f'{model_dir}/C_matrix.npy', C)
        np.save(f'{model_dir}/fi_matrix.npy', fi)
        with open(f'{model_dir}/thresholds.json', 'w') as f:
            json.dump({
                't2': t2_threshold,
                'q': Q_alpha,
                'combined': threshold_combined
            }, f)
    
    return pca, pca_data, D, t2_threshold, C, Q_alpha, fi, threshold_combined

def train_hipca(data, healthy_samples, non_healthy_samples, model_dir, 
               use_ks=True, ks_method='asymp', ks_pval=0.001, only_nonhealthy=False):
    """
    Train hiPCA model
    
    Args:
        data: Raw microbiome data
        healthy_samples: List of healthy sample IDs
        non_healthy_samples: List of non-healthy sample IDs
        model_dir: Directory to save model outputs
        use_ks: Whether to perform KS test feature selection
        ks_method: KS test method
        ks_pval: KS test p-value threshold
        only_nonhealthy: Use only non-healthy associated features
        
    Returns:
        Tuple of model components
    """
    # Feature selection
    features = []
    if use_ks:
        healthy_feats, nonhealthy_feats = ks_test(
            data, healthy_samples, non_healthy_samples, 
            method=ks_method, p_val=ks_pval)
        features = nonhealthy_feats if only_nonhealthy else healthy_feats + nonhealthy_feats
    
    # Data transformation
    transformed_data, scaler = transform_and_scale(
        data[healthy_samples], features, model_dir)
    
    # PCA modeling
    pca, pca_data, D, t2, C, Q, fi, combined = calculate_pca_model(
        transformed_data, model_dir=model_dir)
    
    print(f'Thresholds - T²: {t2:.2f}, Q: {Q:.2f}, Combined: {combined:.2f}')
    
    return features, pca, pca_data, D, t2, C, Q, fi, combined, transformed_data

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Train hiPCA model")
    parser.add_argument('--model_name', required=True, help="Name for the model")
    parser.add_argument('--input', required=True, help="Path to input data")
    parser.add_argument('--metadata', required=True, help="Path to metadata")
    parser.add_argument('--sample_col', required=True, help="Column with sample IDs")
    parser.add_argument('--diagnosis_col', required=True, help="Column with diagnoses")
    parser.add_argument('--control_label', required=True, help="Label for control samples")
    parser.add_argument('--ks_method', default='asymp', 
                       choices=['exact', 'approx', 'asymp'],
                       help='KS test computation method')
    return parser.parse_args()

def main():
    """Main execution function"""
    args = parse_arguments()
    model_dir = f'model_data/{args.model_name}'
    
    # Load data
    data = pd.read_csv(args.input, sep='\t', index_col=0)
    metadata = pd.read_csv(args.metadata, sep='\t')
    
    # Get sample groups
    healthy = metadata[metadata[args.diagnosis_col] == args.control_label][args.sample_col]
    non_healthy = metadata[metadata[args.diagnosis_col] != args.control_label][args.sample_col]
    
    # Train model
    train_hipca(
        data=data,
        healthy_samples=healthy,
        non_healthy_samples=non_healthy,
        model_dir=model_dir,
        use_ks=True,
        ks_method=args.ks_method,
        only_nonhealthy=True
    )

if __name__ == "__main__":
    main()