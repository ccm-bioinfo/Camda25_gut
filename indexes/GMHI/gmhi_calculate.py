import pandas as pd
import numpy as np
import argparse
import pickle

def get_Psi(set_M, sample):
    """
    Calculate the Psi value for a given marker set and sample.
    
    Args:
        set_M: Set of marker species (either healthy or unhealthy markers)
        sample: Pandas Series containing species abundance data for a sample
    
    Returns:
        Psi: Calculated Psi value used in GMHI computation
    """
    # Find intersection of markers present in sample (non-zero abundance) and marker set
    M_in_sample = set(sample[sample != 0].index) & set_M
    
    # Calculate relative abundance of markers in sample vs total markers in set
    R_M_sample = np.divide(len(M_in_sample), len(set_M))
    
    # Get abundance values for markers present in sample
    n = sample[sample != 0][set(M_in_sample)]
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
    samples = tax.columns 
    GMHI = pd.Series(index=samples, name='GMHI', dtype='float64')
    
    for sample in samples:
        # Calculate Psi for healthy markers
        Psi_MH = get_Psi(MH, tax[sample])
        # Calculate Psi for unhealthy markers
        Psi_MN = get_Psi(MN, tax[sample])
        
        # GMHI is log10 ratio of healthy to unhealthy Psi values
        GMHI_sample = np.log10(np.divide(Psi_MH, Psi_MN))
        GMHI[sample] = GMHI_sample
        
    return GMHI 

def main():
    """
    Main function to process microbiome data and calculate GMHI scores.
    Supports both taxonomy-only analysis and integrated pathway analysis.
    """
    # Global thresholds (currently unused but may be for future functionality)
    global theta_f, theta_d
    theta_f = 1.4
    theta_d = 0.1

    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description="Calculate Gut Microbiome Health Index (GMHI) from taxonomic and pathway data.")
    parser.add_argument("--path", required=True, type=str, help="Path to the model directory containing marker species files") 
    parser.add_argument("--taxonomy", required=True, type=str, help="Path to the taxonomy input file (tab-separated)")
    parser.add_argument("--pathways", type=str, help="Path to the pathways input file (tab-separated, optional)")
    parser.add_argument("--outdir", required=True, type=str, help="Path to the output directory")
    args = parser.parse_args()

    # Branch 1: Taxonomy-only analysis (when pathways argument is not provided)
    if args.pathways == None:
        # Load taxonomic marker species (healthy and unhealthy)
        taxonomy = pd.read_csv(f"{args.path.replace('/', '')}/taxonomy.csv")
        MH_tax, MN_tax = set(taxonomy['Healthy']), set(taxonomy['Unhealthy'])
        
        # Load taxonomic abundance data
        tax_data = pd.read_csv(args.taxonomy, sep="\t", index_col=0)
        
        # Calculate GMHI for all samples
        tax_gmhi = get_all_GMHI(tax_data, MH_tax, MN_tax)
        
        # Create results dataframe
        results = pd.DataFrame()
        results['GMHI'] = list(tax_gmhi)
        # Classify samples as healthy (GMHI > 0) or unhealthy (GMHI <= 0)
        results['Prediction'] = ['Healthy' if x > 0 else 'Unhealthy' for x in list(tax_gmhi)]
        results.index = tax_gmhi.index
        
        # Save results
        results.to_csv(args.outdir.replace('/', '') + '/GMHI_results.csv')
    
    # Branch 2: Integrated analysis with pathways data
    else:
        # Load all marker species sets (integrated pathways, unintegrated pathways, taxonomy)
        integrated = pd.read_csv(f"{args.path.replace('/', '')}/pathways_integrated.csv")
        unintegrated = pd.read_csv(f"{args.path.replace('/', '')}/pathways_unintegrated.csv")
        taxonomy = pd.read_csv(f"{args.path.replace('/', '')}/taxonomy.csv")

        # Create marker sets for each data type
        MH_in, MN_in = set(integrated['Healthy']), set(integrated['Unhealthy'])
        MH_un, MN_un = set(unintegrated['Healthy']), set(unintegrated['Unhealthy'])
        MH_tax, MN_tax = set(taxonomy['Healthy']), set(taxonomy['Unhealthy'])

        # Load input data
        tax_data = pd.read_csv(args.taxonomy, sep="\t", index_col=0)
        paths_data = pd.read_csv(args.pathways, sep="\t", index_col=0)

        # Separate integrated and unintegrated pathway data
        unintegrated_data = paths_data.T[[x for x in paths_data.index if 'UNINTEGRATED' in x]]
        integrated_data = paths_data.T[[x for x in paths_data.index if 'UNINTEGRATED' not in x]]

        # Calculate GMHI for each data type
        tax_gmhi = get_all_GMHI(tax_data, MH_tax, MN_tax)
        un_gmhi = get_all_GMHI(unintegrated_data.T, MH_un, MN_un)
        in_gmhi = get_all_GMHI(integrated_data.T, MH_in, MN_in)

        # Combine all GMHI scores into a single dataframe
        new_data = pd.DataFrame(zip(un_gmhi, in_gmhi, tax_gmhi), 
                               columns=['GMHI_unintegrated', 'GMHI_integrated', 'GMHI_taxonomy'])
        new_data.index = tax_data.columns

        # Load pre-trained random forest model and make predictions
        with open(f"{args.path.replace('/', '')}/rf_gmhi.pkl", 'rb') as f:
            rf_gmhi_model = pickle.load(f)

        # Use the three GMHI scores as features for RF prediction
        new_data['RF GMHI prediction'] = rf_gmhi_model.predict(new_data)

        # Save results
        new_data.to_csv(args.outdir.replace('/', '') + '/RFGMHI_results.csv')


if __name__ == "__main__":
    main()