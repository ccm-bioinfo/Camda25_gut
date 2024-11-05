import pandas as pd
import numpy as np
import argparse
import pickle

def get_Psi(set_M,sample):
    M_in_sample=set(sample[sample!=0].index) & set_M
    

    R_M_sample=np.divide(len(M_in_sample),len(set_M))
    

    n=sample[sample!=0][set(M_in_sample)]
    log_n=np.log(n)
    sum_nlnn=np.sum(n*log_n)
    

    Psi=np.divide(R_M_sample,len(set_M))*np.absolute(sum_nlnn)
    

    if Psi==0:
        Psi=1e-05
    return Psi

def get_all_GMHI(tax,MH,MN):
    samples=tax.columns 
    GMHI=pd.Series(index=samples,name='GMHI',dtype='float64')
    for sample in samples:
        

        Psi_MH=get_Psi(MH,tax[sample])
        Psi_MN=get_Psi(MN,tax[sample])
        
        GMHI_sample=np.log10(np.divide(Psi_MH,Psi_MN))
        GMHI[sample]=GMHI_sample
        
    return GMHI 

def main():
    global theta_f, theta_d
    theta_f = 1.4
    theta_d = 0.1

    parser = argparse.ArgumentParser(description="A script that requires --name and --path arguments.")
    parser.add_argument("--path", required=True, type=str, help="Path to the model directory") 
    parser.add_argument("--taxonomy", required=True, type=str, help="Path to the taxonomy input file")
    parser.add_argument("--pathways", type=str, help="Path to the pathways input file")
    parser.add_argument("--outdir", required=True, type=str, help="Path to the output directory")
    args = parser.parse_args()

    if args.pathways == None:
        taxonomy = pd.read_csv(f"{args.path.replace('/', '')}/taxonomy.csv")
        MH_tax, MN_tax = set(taxonomy['Healthy']), set(taxonomy['Unhealthy'])
        tax_data = pd.read_csv(args.taxonomy, sep="\t", index_col = 0)
        tax_gmhi = get_all_GMHI(tax_data, MH_tax, MN_tax)
        results = pd.DataFrame()
        results['GMHI'] = list(tax_gmhi)
        results['Prediction'] = ['Healthy' if x> 0 else 'Unhealthy' for x in list(tax_gmhi)]
        results.index = tax_gmhi.index
        results.to_csv(args.outdir.replace('/', '') +'/GMHI_results.csv')
    
    else:

        integrated = pd.read_csv(f"{args.path.replace('/', '')}/pathways_integrated.csv")
        unintegrated = pd.read_csv(f"{args.path.replace('/', '')}/pathways_unintegrated.csv")
        taxonomy = pd.read_csv(f"{args.path.replace('/', '')}/taxonomy.csv")

        MH_in, MN_in = set(integrated['Healthy']), set(integrated['Unhealthy'])
        MH_un, MN_un = set(unintegrated['Healthy']), set(unintegrated['Unhealthy'])
        MH_tax, MN_tax = set(taxonomy['Healthy']), set(taxonomy['Unhealthy'])

        tax_data = pd.read_csv(args.taxonomy, sep="\t", index_col = 0)
        paths_data = pd.read_csv(args.pathways, sep="\t", index_col = 0)

        unintegrated_data = paths_data.T[[x for x in paths_data.index if 'UNINTEGRATED' in x]]
        integrated_data = paths_data.T[[x for x in paths_data.index if 'UNINTEGRATED' not in x]]

        tax_gmhi = get_all_GMHI(tax_data, MH_tax, MN_tax)
        un_gmhi = get_all_GMHI(unintegrated_data.T, MH_un, MN_un)
        in_gmhi = get_all_GMHI(integrated_data.T, MH_in, MN_in)

        new_data = pd.DataFrame(zip(un_gmhi, in_gmhi, tax_gmhi), columns = ['GMHI_unintegrated', 'GMHI_integrated', 'GMHI_taxonomy'])

        new_data.index = tax_data.columns

        with open(f"{args.path.replace('/', '')}/rf_gmhi.pkl", 'rb') as f:
            rf_gmhi_model = pickle.load(f)

        new_data['RF GMHI prediction'] = rf_gmhi_model.predict(new_data)

        new_data.to_csv(args.outdir.replace('/', '') +'/RFGMHI_results.csv')


if __name__ == "__main__":
    main()