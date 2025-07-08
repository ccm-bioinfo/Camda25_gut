"""
hiPCA Evaluation Script
A streamlined implementation for calculating health indexes using a pre-trained hiPCA model
"""

import pandas as pd
import numpy as np
import pickle
import joblib
import json
import argparse
from pathlib import Path

class HiPCAEvaluator:
    """Class for evaluating samples using a pre-trained hiPCA model"""
    
    def __init__(self, model_path):
        """
        Initialize evaluator with model path
        
        Args:
            model_path: Path to directory containing hiPCA model files
        """
        self.model_path = Path(model_path)
        self._load_model_components()
        
    def _load_model_components(self):
        """Load all required model components from disk"""
        # Load scaling parameters and scaler
        self.scaling_data = pd.read_csv(self.model_path / 'scaling_parameters.csv')
        self.features = list(self.scaling_data['specie'])
        
        with open(self.model_path / 'scaler.pkl', 'rb') as f:
            self.scaler = pickle.load(f)
            
        # Load PCA model and matrices
        self.pca = joblib.load(self.model_path / 'pca_model.pkl')
        self.D = np.load(self.model_path / 'D_matrix.npy')
        self.C = np.load(self.model_path / 'C_matrix.npy')
        self.fi = np.load(self.model_path / 'fi_matrix.npy')
        
        # Load thresholds
        with open(self.model_path / 'thresholds.json', 'r') as f:
            thresholds = json.load(f)
            self.t2_threshold = thresholds['t2']
            self.Q_alpha = thresholds['c']
            self.threshold_combined = thresholds['combined']
    
    @staticmethod
    def custom_transform(x):
        """Custom abundance transformation function"""
        return np.log2(2 * x + 1e-5) if x <= 1 else np.sqrt(x)
    
    def transform_data(self, raw_data):
        """
        Transform and scale input data using model parameters
        
        Args:
            raw_data: Raw microbiome abundance DataFrame
            
        Returns:
            Transformed and scaled DataFrame
        """
        # Create dataframe with model features
        data = pd.DataFrame()
        for feature in self.features:
            data[feature] = raw_data.T[feature] if feature in raw_data.index else 0
        
        # Apply transformations
        transformed = data.applymap(self.custom_transform)
        scaled = self.scaler.transform(transformed)
        
        return pd.DataFrame(scaled, columns=transformed.columns, index=raw_data.T.index)
    
    def calculate_indexes(self, data):
        """
        Calculate health indexes for transformed data
        
        Args:
            data: Transformed microbiome data
            
        Returns:
            DataFrame with index values and predictions
        """
        try:
            projected = self.pca.transform(data)
        except:
            projected = np.array(data)
        
        results = []
        for sample in projected:
            # Calculate all three indexes
            t2 = sample.T @ self.D @ sample
            q = sample.T @ self.C @ sample
            combined = sample.T @ self.fi @ sample
            
            # Make predictions
            pred_t2 = 'Healthy' if t2 <= self.t2_threshold else 'Unhealthy'
            pred_q = 'Healthy' if q <= self.Q_alpha else 'Unhealthy'
            pred_combined = 'Healthy' if combined <= self.threshold_combined else 'Unhealthy'
            
            results.append({
                'T2': t2,
                'Prediction T2': pred_t2,
                'Q': q,
                'Prediction Q': pred_q,
                'Combined Index': combined,
                'Combined Prediction': pred_combined
            })
        
        return pd.DataFrame(results, index=data.index)

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(
        description="Evaluate samples using a pre-trained hiPCA model"
    )
    parser.add_argument(
        "--path", 
        required=True, 
        help="Path to the model directory"
    )
    parser.add_argument(
        "--input", 
        required=True, 
        help="Path to the input data file"
    )
    parser.add_argument(
        "--outdir", 
        required=True, 
        help="Path to the output directory"
    )
    args = parser.parse_args()

    # Initialize evaluator
    evaluator = HiPCAEvaluator(args.path)
    
    # Load and process data
    raw_data = pd.read_csv(args.input, sep='\t', index_col=0)
    transformed_data = evaluator.transform_data(raw_data)
    
    # Calculate indexes and save results
    results = evaluator.calculate_indexes(transformed_data)
    results.to_csv(Path(args.outdir) / 'hiPCA_results.csv')

if __name__ == "__main__":
    main()