import pandas as pd
import numpy as np
import json
import argparse
from statistics import mean
from scipy.stats import kstest, chi2, norm
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import IsolationForest
import warnings

warnings.filterwarnings("ignore")


def ks_test(df, healthy, non_healthy, method_ks='asymp', p_val=0.001):
    """
    Perform Kolmogorov-Smirnov test to identify features that differ between healthy and non-healthy samples.
    
    Parameters:
    - df: DataFrame with features as rows and samples as columns
    - healthy: List of healthy sample IDs
    - non_healthy: List of non-healthy sample IDs
    - method_ks: Method for KS test computation ('asymp', 'exact', 'approx')
    - p_val: p-value threshold for feature selection
    
    Returns:
    - healthy_features: Features enriched in healthy samples
    - nonhealthy_features: Features enriched in non-healthy samples
    """
    # Transpose data to have samples as rows for each group
    healthy_df = df[[x for x in df.columns if x in healthy]].T
    nonhealthy_df = df[[x for x in df.columns if x in non_healthy]].T
    
    healthy_features = []
    nonhealthy_features = []
    
    # Test each feature for differential abundance
    for feature in list(df.index):
        # Test if healthy samples have lower values than non-healthy
        if kstest(list(healthy_df[feature]), list(nonhealthy_df[feature]), 
                 alternative='less', method=method_ks).pvalue <= p_val:
            healthy_features.append(feature)
        
        # Test if non-healthy samples have lower values than healthy
        if kstest(list(nonhealthy_df[feature]), list(healthy_df[feature]), 
                 alternative='less', method=method_ks).pvalue <= p_val:
            nonhealthy_features.append(feature)
    
    print(f'# Healthy features selected by KS: {len(healthy_features)}')
    print(f'# Unhealthy features selected by KS: {len(nonhealthy_features)}')
    
    return healthy_features, nonhealthy_features


def custom_transform(x):
    """
    Apply custom transformation to normalize data values.
    
    Parameters:
    - x: Input value
    
    Returns:
    - Transformed value using log2 for values <= 1, sqrt for values > 1
    """
    if x <= 1:
        return np.log2(2 * x + 0.00001)
    else:
        return np.sqrt(x)


def transform_data(df, features):
    """
    Transform and standardize the selected features.
    
    Parameters:
    - df: Input DataFrame
    - features: List of features to include
    
    Returns:
    - selected2: Transformed and standardized DataFrame
    - scaling_data: Scaling parameters for each feature
    - scaler: Fitted StandardScaler object
    """
    scaler = StandardScaler()
    aux = pd.DataFrame()
    
    # Create dataframe with selected features, filling missing with zeros
    for item in list(set(features)):
        if item in df.index:
            aux[item] = list(df.T[item])
        else:
            aux[item] = [0 for x in range(len(df.T))]
    
    # Apply custom transformation to all values
    selected = aux.applymap(custom_transform)
    
    # Fit and transform using StandardScaler
    scaler.fit(selected)
    selected2 = scaler.transform(selected)
    
    # Store scaling parameters
    scaling_data = pd.DataFrame(
        zip(selected.columns, scaler.mean_, scaler.scale_), 
        columns=['specie', 'mean', 'std']
    )
    
    # Convert back to DataFrame with proper indexing
    selected2 = pd.DataFrame(selected2, columns=selected.columns)
    selected2.index = df.T.index
    
    return selected2, scaling_data, scaler


def calculate_pca_stats(df, variance_for_pc=0.9, alpha=0.05):
    """
    Calculate PCA statistics and control limits for T² and Q statistics.
    
    Parameters:
    - df: Input DataFrame
    - variance_for_pc: Variance threshold for selecting principal components
    - alpha: Significance level for control limits
    
    Returns:
    - pca: Fitted PCA object
    - pca_data: DataFrame with PCA statistics
    - D: Matrix for T² calculation
    - t2_threshold: Control limit for T² statistic
    - C: Matrix for Q calculation
    - Q_alpha: Control limit for Q statistic
    - fi: Combined index matrix
    - threshold_combined: Control limit for combined index
    """
    pca = PCA()
    pca.fit(df)
    
    # Extract PCA components and statistics
    eigenvalues = pca.explained_variance_
    eigenvectors = pca.components_
    singular = pca.singular_values_
    
    # Create DataFrame with PCA statistics
    pca_data = pd.DataFrame(
        zip(eigenvectors, eigenvalues, singular), 
        columns=('Eigenvectors', 'Explained_variance', 'Singular_values')
    ).sort_values('Explained_variance', ascending=False)
    
    # Calculate variance percentages
    pca_data['%variance'] = pca_data['Explained_variance'] / sum(pca_data['Explained_variance'])
    pca_data = pca_data.sort_values('%variance', ascending=False)
    pca_data['%variance_cumulative'] = pca_data['%variance'].cumsum()
    
    # Select principal components based on variance threshold
    principal_components = list(pca_data[pca_data['%variance_cumulative'] < variance_for_pc]['Eigenvectors'])
    print(f'# Principal Components selected: {len(principal_components)}')
    
    # Calculate T² statistic parameters
    principal_values = list(pca_data[pca_data['%variance_cumulative'] < variance_for_pc]['Explained_variance'])
    D = np.array(principal_components).T @ np.linalg.inv(np.diag(principal_values)) @ np.array(principal_components)
    deg_free = len(principal_components)
    t2_threshold = chi2.ppf(1-alpha, deg_free)
    
    # Calculate Q statistic parameters
    principal_components_residual = list(pca_data[pca_data['%variance_cumulative'] >= variance_for_pc]['Eigenvectors'])
    principal_values_residual = list(pca_data[pca_data['%variance_cumulative'] >= variance_for_pc]['Explained_variance'])
    
    C = np.array(principal_components_residual).T @ np.array(principal_components_residual)
    Theta1 = sum(principal_values_residual)
    Theta2 = sum([x**2 for x in principal_values_residual])
    Theta3 = sum([x**3 for x in principal_values_residual])
    
    # Calculate Q control limit using Box approximation
    c_alpha = norm.ppf(1-alpha)
    h0 = 1-((2*Theta1*Theta3)/(3*(Theta2**2)))
    
    # Multiple formulations for Q_alpha calculation (keeping original approach)
    Q_alpha = Theta1*(((((c_alpha*np.sqrt(2*Theta2*(h0**2)))/Theta1)+1+((Theta2*h0*(h0-1))/(Theta1**2))))**(1/h0))
    Q_alpha = Theta1*(((((np.sqrt(c_alpha*(2*Theta2*(h0**2))))/Theta1)+1-((Theta2*h0*(h0-1))/(Theta1**2))))**(1/h0))
    Q_alpha = (Theta2/Theta1) * chi2.ppf(alpha, len(principal_components_residual)) * ((Theta1**2)/Theta2)
    
    # Calculate combined index parameters
    fi = D/t2_threshold + C/Q_alpha
    g = ((len(principal_components) / t2_threshold**2) + (Theta2 / Q_alpha**2)) / ((len(principal_components)/t2_threshold) + (Theta1 / Q_alpha))
    h = ((len(principal_components)/t2_threshold) + (Theta1 / Q_alpha))**2 / ((len(principal_components) / t2_threshold**2) + (Theta2 / Q_alpha**2))
    
    chi_value = chi2.ppf(1-alpha, h)
    threshold_combined = g*chi_value
    
    return pca, pca_data, D, t2_threshold, C, Q_alpha, fi, threshold_combined


def hiPCA(df, healthy, non_healthy, features=[], ks=False, method='auto', p_val=0.001, only_nonhealthy_features=False):
    """
    Main hiPCA function that performs feature selection and PCA analysis.
    
    Parameters:
    - df: Input DataFrame
    - healthy: List of healthy sample IDs
    - non_healthy: List of non-healthy sample IDs
    - features: Pre-selected features (if any)
    - ks: Whether to perform KS test for feature selection
    - method: Method for KS test
    - p_val: p-value threshold for KS test
    - only_nonhealthy_features: Whether to use only non-healthy features
    
    Returns:
    - Multiple objects needed for hiPCA analysis
    """
    # Perform KS test if requested
    if ks:
        healthy_features, non_healthy_features = ks_test(df, healthy, non_healthy, method_ks=method, p_val=p_val)
        features = healthy_features + non_healthy_features
    
    # Transform data using selected features
    selected, scaling_data, scaler = transform_data(df[[x for x in healthy if x in df.columns]], features)
    
    # Calculate PCA statistics and control limits
    pca, pca_data, D, t2_threshold, C, Q_alpha, fi, threshold_combined = calculate_pca_stats(selected)
    
    print(t2_threshold, Q_alpha, threshold_combined)
    
    return features, pca, pca_data, D, t2_threshold, C, Q_alpha, fi, threshold_combined, selected, scaling_data, scaler


def transform_data_evaluate(df, scaling_data, scaler):
    """
    Transform new data using previously fitted scaler and parameters.
    
    Parameters:
    - df: Input DataFrame to transform
    - scaling_data: Scaling parameters from training
    - scaler: Fitted StandardScaler object
    
    Returns:
    - selected2: Transformed DataFrame
    """
    features = list(scaling_data['specie'])
    aux = pd.DataFrame()
    
    # Create dataframe with required features
    for item in list(set(features)):
        if item in df.index:
            aux[item] = list(df.T[item])
        else:
            aux[item] = [0 for x in range(len(df.T))]
    
    # Apply custom transformation and scaling
    selected = aux.applymap(custom_transform)
    selected = selected[features]
    selected2 = scaler.transform(selected)
    selected2 = pd.DataFrame(selected2, columns=selected.columns)
    selected2.index = df.T.index
    
    return selected2


def calculate_index(data_transformed, C, D, fi, pca, t2_threshold, Q_alpha, threshold_combined):
    """
    Calculate T², Q, and combined indices for anomaly detection.
    
    Parameters:
    - data_transformed: Transformed input data
    - C, D, fi: Matrices for index calculations
    - pca: Fitted PCA object
    - t2_threshold, Q_alpha, threshold_combined: Control limits
    
    Returns:
    - DataFrame with sample IDs, indices, and predictions
    """
    T2, Q, combined = [], [], []
    pred_t2, pred_Q, pred_combined = [], [], []
    
    # Calculate indices for each sample
    try:
        # Use PCA transformation
        for item in pca.transform(data_transformed):
            # Calculate T² statistic
            index = item.T @ D @ item
            T2.append(index)
            pred_t2.append('Unhealthy' if index > t2_threshold else 'Healthy')
            
            # Calculate Q statistic
            index2 = item.T @ C @ item
            Q.append(index2)
            pred_Q.append('Unhealthy' if index2 > Q_alpha else 'Healthy')
            
            # Calculate combined index
            index3 = item.T @ fi @ item
            combined.append(index3)
            pred_combined.append('Unhealthy' if index3 > threshold_combined else 'Healthy')
    except:
        # Fallback to direct array processing
        for item in np.array(data_transformed):
            # Calculate T² statistic
            index = item.T @ D @ item
            T2.append(index)
            pred_t2.append('Unhealthy' if index > t2_threshold else 'Healthy')
            
            # Calculate Q statistic
            index2 = item.T @ C @ item
            Q.append(index2)
            pred_Q.append('Unhealthy' if index2 > Q_alpha else 'Healthy')
            
            # Calculate combined index
            index3 = item.T @ fi @ item
            combined.append(index3)
            pred_combined.append('Unhealthy' if index3 > threshold_combined else 'Healthy')
    
    return pd.DataFrame(
        zip(data_transformed.index, T2, pred_t2, Q, pred_Q, combined, pred_combined), 
        columns=['SampleID', 'T2', 'Prediction T2', 'Q', 'Prediction Q', 'Combined Index', 'Combined Prediction']
    )


def parse_args():
    """
    Parse command line arguments.
    
    Returns:
    - Parsed arguments object
    """
    parser = argparse.ArgumentParser(description="Train hiPCA model")
    parser.add_argument('--input', required=True, type=str, help="Path to input data")
    parser.add_argument('--metadata', required=True, type=str, help="Path to metadata")
    parser.add_argument('--sample', required=True, type=str, help="Name of the column containing the ID of the sample in the metadata")
    parser.add_argument('--diagnosis', required=True, type=str, help="Name of the column containing the diagnosis in the metadata")
    parser.add_argument('--control', required=True, type=str, help="Identifier of the control sample")
    parser.add_argument('--method', type=str, choices=['exact', 'approx', 'asymp'], default='asymp', help='Computation method for KS test to use')
    
    return parser.parse_args()


def main():
    """
    Main function that orchestrates the hiPCA analysis workflow.
    """
    args = parse_args()
    
    # Load data and metadata
    data = pd.read_csv(f'{args.input}', sep='\t', index_col=0)
    metadata = pd.read_csv(f'{args.metadata}', sep='\t')
    
    # Filter metadata to match available samples
    metadata = metadata[metadata[args.sample].isin(data.columns)]
    
    # Define disease condition categories
    conditions = dict()
    conditions['ID'] = ['acute_diarrhoea', 'STH', 'respiratoryinf']
    conditions['NP'] = ['adenoma', 'adenoma;hypercholesterolemia', 'adenoma;hypercholesterolemia;metastases',
                       'adenoma;hypertension', 'adenoma;metastases', 'CRC', 'CRC;hypercholesterolemia',
                       'CRC;hypercholesterolemia;hypertension', 'CRC;hypertension', 'CRC;metastases',
                       'CRC;T2D', 'few_polyps', 'T2D;adenoma', 'recipient_before']
    conditions['MD'] = ['hypercholesterolemia', 'IGT', 'IGT;MS', 'metabolic_syndrome', 'T2D', 'T2D;respiratoryinf']
    conditions['MBD'] = ['schizophrenia', 'ME/CFS', 'PD']
    conditions['CD'] = ['ACVD', 'CAD', 'CAD;T2D', 'HF;CAD', 'HF;CAD;T2D', 'hypertension', 'BD']
    conditions['DD'] = ['CD', 'cirrhosis', 'IBD', 'UC']
    conditions['ALL'] = [x for x in metadata[args.diagnosis].unique().tolist() if x != args.control]
    
    # Identify outliers in healthy samples using Isolation Forest
    metadata_healthy = metadata[metadata[args.diagnosis] == args.control]
    data_healthy = data[list(metadata_healthy['sample'])]
    iso = IsolationForest(random_state=42, contamination='auto')
    iso.fit(data_healthy.T)
    
    # Get anomaly scores and sort by informativeness
    scores = iso.decision_function(data_healthy.T)
    X_majority_df = data_healthy.T.copy()
    X_majority_df['score'] = scores
    X_majority_sorted = X_majority_df.sort_values(by='score')
    
    # Process each condition separately
    for condition in conditions:
        print(f"\n=== Processing condition: {condition} ===")
        
        # Get samples for current condition
        metadata_aux2 = metadata[metadata['category'].isin(conditions[condition])]
        n_minority = len(metadata_aux2)
        
        # Select balanced number of healthy controls (2:1 ratio)
        n_majority_target = 2 * n_minority if 2 * n_minority < len(metadata_healthy) else len(metadata_healthy)
        X_majority_selected = X_majority_sorted.iloc[:n_majority_target].drop(columns='score')
        healthy_indexes = X_majority_selected.index
        
        # Create balanced dataset
        metadata_aux = metadata[metadata['sample'].isin(list(metadata_aux2['sample']) + list(healthy_indexes))]
        metadata_aux.reset_index(drop=True, inplace=True)
        data_aux = data[list(metadata_aux['sample'])]
        
        # Perform 5-fold cross-validation
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        metadata_aux['fold'] = -1
        
        # Assign fold numbers
        for fold, (_, val_idx) in enumerate(skf.split(metadata_aux, metadata_aux[args.diagnosis])):
            metadata_aux.loc[val_idx, 'fold'] = fold
        
        # Initialize results storage
        t2_results = []
        q_results = []
        combined_results = []
        results_final = []
        
        # Process each fold
        for fold in metadata_aux['fold'].unique():
            print(f"Processing fold {fold}")
            
            # Split data into training and evaluation sets
            metadata_train = metadata_aux[metadata_aux['fold'] != fold]
            samples_train = list(metadata_train[args.sample])
            metadata_evaluate = metadata_aux[metadata_aux['fold'] == fold]
            samples_evaluate = list(metadata_evaluate[args.sample])
            
            data_train = data_aux[samples_train]
            data_evaluate = data_aux[samples_evaluate]
            
            # Identify healthy and non-healthy samples in training set
            healthy = list(metadata_train[metadata_train[args.diagnosis] == args.control][args.sample])
            non_healthy = list(metadata_train[metadata_train[args.diagnosis] != args.control][args.sample])
            
            # Train hiPCA model
            features, pca, pca_data, D, t2_threshold, C, Q_alpha, fi, threshold_combined, selected, scaling, scaler = hiPCA(
                data_train, healthy, non_healthy, ks=True, method=args.method, only_nonhealthy_features=True
            )
            
            # Save scaling parameters
            scaling.to_csv(f'output_aux/{condition}_bacteria_{fold}.csv')
            
            # Transform evaluation data and calculate indices
            data_evaluate = transform_data_evaluate(data_evaluate, scaling, scaler)
            results = calculate_index(data_evaluate, C, D, fi, pca, t2_threshold, Q_alpha, threshold_combined)
            
            # Normalize results by subtracting thresholds
            results['T2'] = results['T2'] - t2_threshold
            results['Q'] = results['Q'] - Q_alpha
            results['Combined Index'] = results['Combined Index'] - threshold_combined
            
            results_final.append(results)
            
            # Calculate F1 scores
            t2_preds = [1 if x == 'Healthy' else 0 for x in results['Prediction T2']]
            q_preds = [1 if x == 'Healthy' else 0 for x in results['Prediction Q']]
            combined_preds = [1 if x == 'Healthy' else 0 for x in results['Combined Prediction']]
            
            true_labels = [1 if x == 'healthy' else 0 for x in metadata_evaluate[args.diagnosis]]
            
            t2_result = f1_score(true_labels, t2_preds, pos_label=0)
            q_result = f1_score(true_labels, q_preds, pos_label=0)
            combined_result = f1_score(true_labels, combined_preds, pos_label=0)
            
            t2_results.append(t2_result)
            q_results.append(q_result)
            combined_results.append(combined_result)
            
            print(f"T2 F1: {t2_result}, Q F1: {q_result}, Combined F1: {combined_result}")
        
        # Save fold results
        fold_results = pd.DataFrame(
            zip(t2_results, q_results, combined_results), 
            columns=['T2', 'Q', 'Combined']
        )
        fold_results = fold_results.T
        fold_results.columns = ['Fold1', 'Fold2', 'Fold3', 'Fold4', 'Fold5']
        fold_results['Mean'] = fold_results.mean(axis=1)
        fold_results.to_csv(f'output_aux/{condition}_f1_score_results_taxonomy_corrected_balanced.csv')
        
        # Save all predictions
        results_final2 = pd.concat(results_final, ignore_index=True)
        results_final2.to_csv(f'output_aux/{condition}_camda2025_preds_taxonomy_corrected_balanced.csv', index=False)


if __name__ == "__main__":
    main()