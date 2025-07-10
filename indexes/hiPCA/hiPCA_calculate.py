import pandas as pd
import pickle
import joblib
import numpy as np
import json
import argparse


def custom_transform(x):
    """
    Apply custom transformation to data values.
    
    Args:
        x: Input value to transform
        
    Returns:
        Transformed value using log2 for x <= 1, sqrt for x > 1
    """
    if x <= 1:
        return np.log2(2 * x + 0.00001)
    else:
        return np.sqrt(x)


def transform_data(df):
    """
    Transform input data using scaling parameters and custom transformation.
    
    Args:
        df: Input DataFrame with species data
        
    Returns:
        Transformed and scaled DataFrame ready for PCA analysis
    """
    # Load scaling parameters and feature list
    scaling_data = pd.read_csv(f'{path}/scaling_parameters.csv')
    features = list(scaling_data['specie'])
    
    # Load pre-fitted scaler
    with open(f'{path}/scaler.pkl', 'rb') as file:
        scaler = pickle.load(file)
    
    # Create auxiliary DataFrame with required features
    aux = pd.DataFrame()
    for item in list(set(features)):
        if item in df.index:
            aux[item] = list(df.T[item])
        else:
            # Fill missing features with zeros
            aux[item] = [0 for x in range(len(df.T))]
    
    # Apply custom transformation to all values
    selected = aux.applymap(custom_transform)
    
    # Reorder columns to match training feature order
    selected = selected[features]
    
    # Apply pre-fitted scaler transformation
    selected2 = scaler.transform(selected)
    selected2 = pd.DataFrame(selected2, columns=selected.columns)
    selected2.index = df.T.index
    
    return selected2


def calculate_index(data_transformed):
    """
    Calculate health indices (T2, Q, Combined) using PCA and matrix operations.
    
    Args:
        data_transformed: Transformed DataFrame from transform_data()
        
    Returns:
        DataFrame with sample IDs, calculated indices, and health predictions
    """
    # Load pre-computed matrices and PCA model
    D = np.load(f'{path}/D_matrix.npy')
    C = np.load(f'{path}/C_matrix.npy')
    fi = np.load(f'{path}/fi_matrix.npy')
    pca = joblib.load(f'{path}/pca_model.pkl')
    
    # Load threshold values for health classification
    with open(f'{path}/thresholds.json', 'r') as file:
        thresholds = json.load(file)
        t2_threshold = thresholds['t2']
        Q_alpha = thresholds['c']
        threshold_combined = thresholds['combined']
    
    # Initialize result lists
    T2, Q, combined = [], [], []
    pred_t2, pred_Q, pred_combined = [], [], []
    
    # Try PCA transformation first, fall back to direct array if it fails
    try:
        # Apply PCA transformation and calculate indices
        for item in pca.transform(data_transformed):
            # Calculate T2, Q, and combined indices using matrix operations
            index = item.T @ D @ item
            index2 = item.T @ C @ item
            index3 = item.T @ fi @ item
            
            T2.append(index)
            Q.append(index2)
            combined.append(index3)
            
            # Make health predictions based on thresholds
            pred_t2.append('Unhealthy' if index > t2_threshold else 'Healthy')
            pred_Q.append('Unhealthy' if index2 > Q_alpha else 'Healthy')
            pred_combined.append('Unhealthy' if index3 > threshold_combined else 'Healthy')
            
    except:
        # Fallback: use data directly without PCA transformation
        for item in np.array(data_transformed):
            # Calculate indices using matrix operations
            index = item.T @ D @ item
            index2 = item.T @ C @ item
            index3 = item.T @ fi @ item
            
            T2.append(index)
            Q.append(index2)
            combined.append(index3)
            
            # Make health predictions based on thresholds
            pred_t2.append('Unhealthy' if index > t2_threshold else 'Healthy')
            pred_Q.append('Unhealthy' if index2 > Q_alpha else 'Healthy')
            pred_combined.append('Unhealthy' if index3 > threshold_combined else 'Healthy')
    
    # Create results DataFrame
    return pd.DataFrame(
        zip(data_transformed.index, T2, pred_t2, Q, pred_Q, combined, pred_combined),
        columns=['SampleID', 'T2', 'Prediction T2', 'Q', 'Prediction Q', 'Combined Index', 'Combined Prediction']
    )


def calculate_hiPCA(path_, data):
    """
    Main function to calculate hiPCA results for given data.
    
    Args:
        path_: Path to model directory containing required files
        data: Input DataFrame with species abundance data
        
    Returns:
        DataFrame with hiPCA analysis results
    """
    global path
    path = path_
    data = transform_data(data)
    results = calculate_index(data)
    return results


def main():
    """
    Command-line interface for hiPCA analysis.
    Processes input file and saves results to specified output directory.
    """
    # Set up command-line argument parser
    parser = argparse.ArgumentParser(description="hiPCA health index analysis script")
    parser.add_argument("--path", required=True, type=str, help="Path to the model directory")
    parser.add_argument("--input", required=True, type=str, help="Path to the input file")
    parser.add_argument("--outdir", required=True, type=str, help="Path to the output directory")
    args = parser.parse_args()
    
    # Set global path variable
    global path
    path = args.path
    
    # Load input data (tab-separated format expected)
    data = pd.read_csv(args.input, sep='\t', index_col=0)
    
    # Transform data and calculate health indices
    data = transform_data(data)
    results = calculate_index(data)
    
    # Save results to output directory
    results.to_csv(args.outdir.replace('/', '') + '/hiPCA_results.csv')


if __name__ == "__main__":
    main()