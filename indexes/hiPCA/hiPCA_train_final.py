import pandas as pd
import numpy as np
import json
import pickle
import os
from scipy.stats import kstest, chi2, norm
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import argparse


def ks_test(df, healthy, non_healthy, method_ks='asymp', p_val=0.001):
    """
    Perform Kolmogorov-Smirnov test to identify features that differ between healthy and non-healthy samples.
    
    Args:
        df: DataFrame with features as rows and samples as columns
        healthy: List of healthy sample IDs
        non_healthy: List of non-healthy sample IDs
        method_ks: Method for KS test computation ('exact', 'approx', 'asymp')
        p_val: P-value threshold for significance
    
    Returns:
        Tuple of (healthy_features, nonhealthy_features)
    """
    # Transpose data to have samples as rows for each group
    healthy_df = df[[x for x in df.columns if x in healthy]].T
    nonhealthy_df = df[[x for x in df.columns if x in non_healthy]].T
    
    healthy_features = []
    nonhealthy_features = []
    
    # Test each feature for significant differences between groups
    for feature in list(df.index):
        # Test if healthy samples have lower values (alternative='less')
        if kstest(list(healthy_df[feature]), list(nonhealthy_df[feature]), 
                 alternative='less', method=method_ks).pvalue <= p_val:
            healthy_features.append(feature)
        
        # Test if non-healthy samples have lower values
        if kstest(list(nonhealthy_df[feature]), list(healthy_df[feature]), 
                 alternative='less', method=method_ks).pvalue <= p_val:
            nonhealthy_features.append(feature)
    
    print(f'# Healthy features selected by KS: {len(healthy_features)}')
    print(f'# Unhealthy features selected by KS: {len(nonhealthy_features)}')
    
    return healthy_features, nonhealthy_features


def custom_transform(x):
    """
    Custom transformation function for data preprocessing.
    
    Args:
        x: Input value
    
    Returns:
        Transformed value using log2 for x <= 1, sqrt for x > 1
    """
    if x <= 1:
        return np.log2(2 * x + 0.00001)
    else:
        return np.sqrt(x)


def transform_data(df, features):
    """
    Transform and standardize selected features.
    
    Args:
        df: Input DataFrame
        features: List of features to transform
    
    Returns:
        Transformed and standardized DataFrame
    """
    scaler = StandardScaler()
    aux = pd.DataFrame()
    
    # Extract selected features, filling missing ones with zeros
    for item in list(set(features)):
        if item in df.index:
            aux[item] = list(df.T[item])
        else:
            aux[item] = [0 for x in range(len(df.T))]
    
    # Apply custom transformation
    selected = aux.applymap(custom_transform)

    # Fit scaler and save it
    scaler.fit(selected)
    with open(f'model_data/{model_name}/scaler.pkl', 'wb') as file:
        pickle.dump(scaler, file)
    
    # Transform data
    selected2 = scaler.transform(selected)
    
    # Save scaling parameters
    pd.DataFrame(zip(selected.columns, scaler.mean_, scaler.scale_), 
                columns=['specie', 'mean', 'std']).to_csv(f'model_data/{model_name}/scaling_parameters.csv', index=False)

    # Convert back to DataFrame with proper indexing
    selected2 = pd.DataFrame(selected2, columns=selected.columns)
    selected2.index = df.T.index

    return selected2


def calculate_pca_stats(df, variance_for_pc=0.9, alpha=0.05):
    """
    Calculate PCA statistics and control limits for T² and Q statistics.
    
    Args:
        df: Input DataFrame
        variance_for_pc: Variance threshold for principal component selection
        alpha: Significance level for control limits
    
    Returns:
        Tuple of PCA model, PCA data, and various matrices/thresholds
    """
    # Fit PCA model
    pca = PCA()
    pca.fit(df)

    # Save PCA model
    with open(f'model_data/{model_name}/pca_model.pkl', 'wb') as file:
        pickle.dump(pca, file)

    # Extract PCA components and variance information
    eigenvalues = pca.explained_variance_
    eigenvectors = pca.components_
    singular = pca.singular_values_
    
    # Create PCA summary DataFrame
    pca_data = pd.DataFrame(zip(eigenvectors, eigenvalues, singular), 
                           columns=('Eigenvectors', 'Explained_variance', 'Singular_values')).sort_values('Explained_variance', ascending=False)
    pca_data['%variance'] = pca_data['Explained_variance'] / sum(pca_data['Explained_variance'])
    pca_data = pca_data.sort_values('%variance', ascending=False)
    pca_data['%variance_cumulative'] = pca_data['%variance'].cumsum()
    
    # Select principal components based on variance threshold
    principal_components = list(pca_data[pca_data['%variance_cumulative'] < variance_for_pc]['Eigenvectors'])
    print(f'# Principal Components selected: {len(principal_components)}')
    
    # Calculate T² statistic components
    principal_values = list(pca_data[pca_data['%variance_cumulative'] < variance_for_pc]['Explained_variance'])
    D = np.array(principal_components).T @ np.linalg.inv(np.diag(principal_values)) @ np.array(principal_components)
    deg_free = len(principal_components) 
    t2_threshold = chi2.ppf(1-alpha, deg_free)
    
    # Calculate Q statistic components (residual space)
    principal_components_residual = list(pca_data[pca_data['%variance_cumulative'] >= variance_for_pc]['Eigenvectors'])
    principal_values_residual = list(pca_data[pca_data['%variance_cumulative'] >= variance_for_pc]['Explained_variance'])
    
    C = np.array(principal_components_residual).T @ np.array(principal_components_residual)
    Theta1 = sum(principal_values_residual)
    Theta2 = sum([x**2 for x in principal_values_residual])
    Theta3 = sum([x**3 for x in principal_values_residual])
    
    # Calculate Q statistic threshold using different methods
    c_alpha = norm.ppf(1-alpha)
    h0 = 1-((2*Theta1*Theta3)/(3*(Theta2**2)))

    # Multiple approaches for Q_alpha calculation (keeping original logic)
    Q_alpha = Theta1*(((((c_alpha*np.sqrt(2*Theta2*(h0**2)))/Theta1)+1+((Theta2*h0*(h0-1))/(Theta1**2))))**(1/h0))
    Q_alpha = Theta1*(((((np.sqrt(c_alpha*(2*Theta2*(h0**2))))/Theta1)+1-((Theta2*h0*(h0-1))/(Theta1**2))))**(1/h0))
    Q_alpha = (Theta2/Theta1) * chi2.ppf(alpha, len(principal_components_residual)) * ((Theta1**2)/Theta2)
    
    # Calculate combined statistic components
    fi = D/t2_threshold + C/Q_alpha
    g = ((len(principal_components) / t2_threshold**2) + (Theta2 / Q_alpha**2)) / ((len(principal_components)/t2_threshold) + (Theta1 / Q_alpha))
    h = ((len(principal_components)/t2_threshold) + (Theta1 / Q_alpha))**2 / ((len(principal_components) / t2_threshold**2) + (Theta2 / Q_alpha**2))

    chi_value = chi2.ppf(1-alpha, h)
    threshold_combined = g*chi_value
    
    return pca, pca_data, D, t2_threshold, C, Q_alpha, fi, threshold_combined


def hotelling_t2(df, pca, pca_data, variance_for_pc=0.9, alpha=0.05):
    """
    Calculate Hotelling's T² statistic for anomaly detection.
    
    Args:
        df: Input DataFrame
        pca: Fitted PCA model
        pca_data: PCA summary data
        variance_for_pc: Variance threshold for PC selection
        alpha: Significance level
    
    Returns:
        Tuple of D matrix, principal components, results DataFrame, and threshold
    """
    # Get principal components and calculate T² matrix
    principal_components = list(pca_data[pca_data['%variance_cumulative'] < variance_for_pc]['Eigenvectors'])
    print(f'# Principal Components selected: {len(principal_components)}')
    principal_values = list(pca_data[pca_data['%variance_cumulative'] < variance_for_pc]['Explained_variance'])
    D = np.array(principal_components).T @ np.linalg.inv(np.diag(principal_values)) @ np.array(principal_components)
    deg_free = len(principal_components) 
    t2_threshold = chi2.ppf(1-alpha, deg_free)
    
    T2 = []
    pred = []
    
    # Calculate T² statistic for each sample
    try:
        for item in pca.transform(df):
            index = item.T @ D @ item
            T2.append(index)
            if index > t2_threshold:
                pred.append('Unhealthy')
            else:
                pred.append('Healthy')
    except:
        for item in np.array(df):
            index = item.T @ D @ item
            T2.append(index)
            if index > t2_threshold:
                pred.append('Unhealthy')
            else:
                pred.append('Healthy')
    
    hoteling = pd.DataFrame(zip(df.index, T2, pred), columns=['Sample', 'T2', 'Prediction T2'])
    
    return D, principal_components, hoteling, t2_threshold


def Q_statistic(df, pca, pca_data, variance_for_pc=0.9, alpha=0.05):
    """
    Calculate Q statistic (SPE - Squared Prediction Error) for anomaly detection.
    
    Args:
        df: Input DataFrame
        pca: Fitted PCA model
        pca_data: PCA summary data
        variance_for_pc: Variance threshold for PC selection
        alpha: Significance level
    
    Returns:
        Tuple of matrices, theta values, results DataFrame, and threshold
    """
    # Get residual space components
    principal_components_residual = list(pca_data[pca_data['%variance_cumulative'] >= variance_for_pc]['Eigenvectors'])
    principal_values_residual = list(pca_data[pca_data['%variance_cumulative'] >= variance_for_pc]['Explained_variance'])
    
    # Calculate Q statistic components
    C = np.array(principal_components_residual).T @ np.array(principal_components_residual)
    Theta1 = sum(principal_values_residual)
    Theta2 = sum([x**2 for x in principal_values_residual])
    Theta3 = sum([x**3 for x in principal_values_residual])
    
    # Calculate Q threshold using multiple approaches
    c_alpha = norm.ppf(1-alpha)
    h0 = 1-((2*Theta1*Theta3)/(3*Theta2**2))

    # Original formula (keeping all variants as in original code)
    Q_alpha = Theta1*(((((c_alpha*np.sqrt(2*Theta2*(h0**2)))/Theta1)+1+((Theta2*h0*(h0-1))/(Theta1**2))))**(1/h0))
    print(Q_alpha)
    Q_alpha = Theta1*(((((np.sqrt(c_alpha*(2*Theta2*(h0**2))))/Theta1)+1-((Theta2*h0*(h0-1))/(Theta1**2))))**(1/h0))
    print(Q_alpha)
    Q_alpha = (Theta2/Theta1) * chi2.ppf(1-alpha, len(principal_components_residual)) * ((Theta1**2)/Theta2)
    print(Q_alpha)
    
    Q = []
    pred = []
    
    # Calculate Q statistic for each sample
    try:
        for item in pca.transform(df):
            index = item.T @ C @ item
            Q.append(index)
            if index > Q_alpha:
                pred.append('Unhealthy')
            else:
                pred.append('Healthy')
    except:
        for item in np.array(df):
            index = item.T @ C @ item
            Q.append(index)
            if index > Q_alpha:
                pred.append('Unhealthy')
            else:
                pred.append('Healthy')
    
    Q_statistic = pd.DataFrame(zip(df.index, Q, pred), columns=['Sample', 'Q', 'Prediction Q'])
    
    return C, Theta1, Theta2, Q_statistic, Q_alpha


def combined_index(df, D, t2_threshold, principal_components, Q_alpha, Theta1, Theta2, pca, alpha=0.05):
    """
    Calculate combined index statistic for anomaly detection.
    
    Args:
        df: Input DataFrame
        D: T² matrix
        t2_threshold: T² threshold
        principal_components: Selected principal components
        Q_alpha: Q statistic threshold
        Theta1, Theta2: Theta parameters
        pca: Fitted PCA model
        alpha: Significance level
    
    Returns:
        DataFrame with combined index results
    """
    # Calculate combined statistic matrix
    fi = D/t2_threshold + (np.eye(len(principal_components[0])) - (np.array(principal_components).T @ np.array(principal_components)))/Q_alpha
    g = ((len(principal_components) / t2_threshold**2) + (Theta2 / Q_alpha**2)) / ((len(principal_components)/t2_threshold) + (Theta1 / Q_alpha))
    h = ((len(principal_components)/t2_threshold) + (Theta1 / Q_alpha))**2 / ((len(principal_components) / t2_threshold**2) + (Theta2 / Q_alpha**2))

    chi_value = chi2.ppf(1-alpha, h)
    threshold_combined = g*chi_value
    
    combined = []
    pred = []

    # Calculate combined statistic for each sample
    try:
        for item in pca.transform(df):
            index = item.T @ fi @ item
            combined.append(index)
            if index > threshold_combined:
                pred.append('Unhealthy')
            else:
                pred.append('Healthy')
    except:
        for item in np.array(df):
            index = item.T @ fi @ item
            combined.append(index)
            if index > threshold_combined:
                pred.append('Unhealthy')
            else:
                pred.append('Healthy')

    combined = pd.DataFrame(zip(df.index, combined, pred), columns=['Sample', 'Combined', 'Prediction Combined']) 
    return combined


def hiPCA(df, healthy, non_healthy, features=[], ks=False, method='auto', p_val=0.001, only_nonhealthy_features=False):
    """
    Main hiPCA (health-informed PCA) function for anomaly detection.
    
    Args:
        df: Input DataFrame with features as rows and samples as columns
        healthy: List of healthy sample IDs
        non_healthy: List of non-healthy sample IDs
        features: List of features to use (if not using KS test)
        ks: Whether to use Kolmogorov-Smirnov test for feature selection
        method: KS test method
        p_val: P-value threshold for KS test
        only_nonhealthy_features: Whether to use only non-healthy features
    
    Returns:
        Tuple of selected features, PCA model, and various matrices/thresholds
    """
    # Perform KS test if requested
    if ks:
        healthy_features, non_healthy_features = ks_test(df, healthy, non_healthy, method_ks=method, p_val=p_val)
        
    # Select features based on strategy
    if only_nonhealthy_features:
        healthy_features = []
        if ks:
            features = healthy_features + non_healthy_features
        selected = transform_data(df[[x for x in healthy if x in df.columns]], features)
    else:
        if ks:
            features = healthy_features + non_healthy_features
        selected = transform_data(df[[x for x in healthy if x in df.columns]], features)
    
    # Calculate PCA statistics and control limits
    pca, pca_data, D, t2_threshold, C, Q_alpha, fi, threshold_combined = calculate_pca_stats(selected)
    
    # Save matrices for later use
    np.save(f'model_data/{model_name}/D_matrix.npy', D)
    np.save(f'model_data/{model_name}/C_matrix.npy', C)
    np.save(f'model_data/{model_name}/fi_matrix.npy', fi)

    # Save thresholds
    thresholds = {'t2': t2_threshold, 'c': Q_alpha, 'combined': threshold_combined}
    with open(f'model_data/{model_name}/thresholds.json', 'w') as json_file:
        json.dump(thresholds, json_file)

    print(t2_threshold, Q_alpha, threshold_combined)
        
    return features, pca, pca_data, D, t2_threshold, C, Q_alpha, fi, threshold_combined, selected


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train hiPCA model")
    parser.add_argument('--model_name', required=True, type=str, help="Name of the model")
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


def main():
    """Main function to train hiPCA model."""
    args = parse_args()
    global model_name
    model_name = args.model_name
    
    # Create output directory
    os.makedirs(f'model_data/{args.model_name}', exist_ok=True)
    
    # Load data and metadata
    data = pd.read_csv(f'{args.input}', sep='\t', index_col=0)
    metadata = pd.read_csv(f'{args.metadata}', sep='\t')

    # Identify healthy and non-healthy samples
    healthy = list(metadata[metadata[args.diagnosis] == args.control][args.sample])
    non_healthy = list(metadata[metadata[args.diagnosis] != args.control][args.sample])

    # Train hiPCA model
    features, pca, pca_data, D, t2_threshold, C, Q_alpha, fi, threshold_combined, selected = hiPCA(
        data, healthy, non_healthy, ks=True, method=args.method, only_nonhealthy_features=True
    )


if __name__ == "__main__":
    main()