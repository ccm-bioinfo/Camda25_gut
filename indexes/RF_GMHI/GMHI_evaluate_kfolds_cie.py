import pandas as pd
import numpy as np
import json
import pickle
import os
from scipy.stats import kstest
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.stats import chi2, norm
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
import argparse
from statistics import mean
from sklearn.ensemble import IsolationForest

import warnings
warnings.filterwarnings("ignore")

def parse_args():
    parser = argparse.ArgumentParser(description="Train hiPCA model")
    parser.add_argument('--input', required=True, type=str, help="Path to input data")
    parser.add_argument('--metadata', required=True, type=str, help="Path to metadata")
    parser.add_argument('--sample', required=True, type=str, help="Name of the column containning the ID of the sample in the metadata")
    parser.add_argument('--diagnosis', required=True, type=str, help="Name of the column containning the diagnosis in the metadata")
    parser.add_argument('--control', required=True, type=str, help="Identifier of the control sample")

    parser.add_argument(
        '--method',
        type=str,
        choices=['exact', 'approx', 'asymp'],
        default='asymp',
        help='Computation method for KS test to use'
    )

    return parser.parse_args()

def get_fH_fN_dH_dN(meta,tax):
    ######### Recibe los data frames de los metadatos y la taxonomia.
    
    #Se obtienen los id's de las muestras saludables identificadas en los metadatos y después 
    #observamos la taxonomia de las muestras saludables
    healthy_id = meta[meta['category']=='healthy']['sample']
    tax_healthy = tax[healthy_id]
    
    #Se obtienen los id's de muestras no saludables y despues se observa la taxonmia de estas muestras
    no_healthy_id = meta[meta['category']!='healthy']['sample']
    tax_no_healthy = tax[no_healthy_id]
    
    #Se obtienen todas las especies de todas las muestras
    species = tax.index
    
    #Definimos lower para establecer una cota y evitar divisiones entre 0
    lower=1e-05
    
    #Se crea un Data Frame que tendrá las metricas como columnas y a las especies como index
    metrics=pd.DataFrame(index=species,columns=['f_H','f_N','d_H','d_N'])
    
    #Este ciclo obtiene para cada especie m las prevalencias en las muestras saludables p_H y no saludables P_N
    #Posteriormente se  agregan f_H,f_N, d_H y d_N al data frame metric
    for specie in species:
        
        #Se localiza la especie en todas las muestras healthy y se obtiene su presencia absoluta
        specie_in_H=tax_healthy.loc[specie,:]
        abs_pres_H=len(specie_in_H[specie_in_H!=0])
        
        #Se localiza la especie en todas las muestras no-healthy y se obtiene su presencia absoluta
        specie_in_N=tax_no_healthy.loc[specie,:]
        abs_pres_N=len(specie_in_N[specie_in_N!=0])
        
        #Se obtiene PH y PN de la especie, tomando en cuenta que si el resultado es 0, entonces se intercambia por la cota 1e-05
        PH=np.divide(abs_pres_H,len(specie_in_H),out=np.asanyarray(lower),where=(abs_pres_H!=0))
        PN=np.divide(abs_pres_N,len(specie_in_N),out=np.asanyarray(lower),where=(abs_pres_N!=0))
        metrics.loc[specie,:]=[np.divide(PH,PN),np.divide(PN,PH),PH-PN,PN-PH]
    return metrics

######### Regresa un DataFrame en el que para cada especie se obtienen sus metricas f_H,f_N,d_H y d_N

def get_MH_MN(metrics,theta_f,theta_d):
    ######### Recibe el conjunto de metricas para cada especie y los parámetros de comparación
    
    
    #Se obtienen las especies beneficiosas que son mayores a los parametros theta_f y theta_d
    health_species_pass_theta_f=set(metrics[metrics['f_H']>=theta_f].index)
    health_species_pass_theta_d=set(metrics[metrics['d_H']>=theta_d].index)
    
    #Se obtienen las especies dañinas que son mayores a los parametros theta_f y theta_d
    no_health_species_pass_theta_f=set(metrics[metrics['f_N']>=theta_f].index)
    no_health_species_pass_theta_d=set(metrics[metrics['d_N']>=theta_d].index)
    
    #Se definen los conjuntos de las especies beneficiosas y dañinas que superan ambos parámetros
    MH=health_species_pass_theta_f & health_species_pass_theta_d
    MN=no_health_species_pass_theta_f & no_health_species_pass_theta_d
    
    # print('|MH|=', len(MH) )
    # print('|MN|=', len(MN))        
    return MH,MN

######### Regresa los conjuntos de especies identificadas beneficiosas MH y dañinas MN, de acuerdo a los parámetros

def get_Psi(set_M,sample):
    ######### Recibe el conjunto M_H o M_N y la muestra con la presencia relativa de cada especie
    
    
    #M_in_sample es el conjunto M_H o M_N intersección las especies presentes en la muestra i
    M_in_sample=set(sample[sample!=0].index) & set_M
    
    #Se calcula la R_M
    R_M_sample=np.divide(len(M_in_sample),len(set_M))
    
    #Se obtiene el array n, que contiene las abundanicas relativas de las especies presentes de M en la muestra i
    #Posteriormente se calcula el logaritmo y la suma
    n=sample[sample!=0][list(M_in_sample)]
    log_n=np.log(n)
    sum_nlnn=np.sum(n*log_n)
    
    #Finalmente se recupera Psi para la muestra i y el conjunto M
    Psi=np.divide(R_M_sample,len(set_M))*np.absolute(sum_nlnn)
    
    #Se evita que el caso Psi sea igual a 0 para evitar división entre 0 en la siguiente función. 
    if Psi==0:
        Psi=1e-05
    return Psi

######### Regresa el número Psi asociado a la muestra i y  al conjunto M_H o M_N.   

def get_all_GMHI(tax,MH,MN):
    ######### Se ingresa la taxonomia, el conjunto de especies MH y MN.
    

    #Se crea la variable GMHI, una serie de pandas que tiene como indice el nombre de la muestra y como información su indice GMHI.
    #Esta serie se llenará con un ciclo for, que recorre todas las especies
    samples=tax.columns 
    GMHI=pd.Series(index=samples,name='GMHI',dtype='float64')
    for sample in samples:
        
        #Se obtiene Psi_MH y Psi_MN con la función get_Psi
        Psi_MH=get_Psi(MH,tax[sample])
        Psi_MN=get_Psi(MN,tax[sample])
        
        #Se hace el cociente y se evalua en el logaritmo base 10. Posteriormente se agrega la información a la serie GMHI
        GMHI_sample=np.log10(np.divide(Psi_MH,Psi_MN))
        GMHI[sample]=GMHI_sample
        
    return GMHI 

######### Se regresa la serie con el índice GMHI de cada muestra

def get_accuracy2(GMHI,meta):    
    return f1_score([0 if x != 'healthy' else 1 for x in meta], [0 if x < 0 else 1 for x in list(GMHI)], pos_label = 0)


def main():
    args = parse_args()
    # global model_name
    # model_name = args.model_name
    # os.makedirs(f'model_data/{args.model_name}', exist_ok=True)
    #! Missing evaluation of correct data
    data = pd.read_csv(f'{args.input}', sep = '\t', index_col = 0)
    metadata = pd.read_csv(f'{args.metadata}', sep = '\t')

    # data = data[[item for item in data.columns if item not in mismatches]]
    metadata = metadata[metadata[args.sample].isin(data.columns)]


    #TODO para filtrar los datos
    conditions = dict()

    conditions['ID'] = [
        'acute_diarrhoea',
        'STH',
        'respiratoryinf'
    ]

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

    conditions['MD'] = [
        'hypercholesterolemia',
        'IGT',
        'IGT;MS',
        'metabolic_syndrome',
        'T2D',
        'T2D;respiratoryinf'
    ]

    conditions['MBD'] = [
        'schizophrenia',
        'ME/CFS',
        'PD'
    ]

    conditions['CD'] = [
        'ACVD',
        'CAD',
        'CAD;T2D',
        'HF;CAD',
        'HF;CAD;T2D',
        'hypertension',
        'BD'
    ]

    conditions['DD'] = [
        'CD',
        'cirrhosis',
        'IBD',
        'UC'
    ]


    theta_f = 1.4
    # theta_d = 0.1 # Esto lo tengo que relajar porque pone en mucha desventaja especies poco abundantes
    theta_d = 0.0001

    metadata_healthy = metadata[metadata['category'] == 'healthy']
    data_healthy = data[list(metadata_healthy['sample'])]
    iso = IsolationForest(random_state=42, contamination='auto')
    iso.fit(data_healthy.T)
    # Get anomaly scores (higher = more normal, lower = more anomalous)
    scores = iso.decision_function(data_healthy.T)
    X_majority_df = data_healthy.T.copy()
    X_majority_df['score'] = scores
    # Sort by score (lowest = more anomalous → more informative)
    X_majority_sorted = X_majority_df.sort_values(by='score')



    for condition in conditions:
        # metadata_aux = metadata[metadata['category'].isin(conditions[condition] + ['healthy'])]
        metadata_aux2 = metadata[metadata['category'].isin(conditions[condition])]
        n_minority = len(metadata_aux2)
        n_majority_target = 3 * n_minority if 3 * n_minority < len(metadata_healthy) else len(metadata_healthy)

        
        # Keep only the top samples needed to reach 2:1 ratio
        X_majority_selected = X_majority_sorted.iloc[:n_majority_target].drop(columns='score')

        healthy_indexes = X_majority_selected.index
        # print(healthy_indexes)
        metadata_aux = metadata[metadata['sample'].isin(list(metadata_aux2['sample']) + list(healthy_indexes))]
        metadata_aux.reset_index(drop=True, inplace=True)   
        data_aux = data[list(metadata_aux['sample'])]

    # # metadata = metadata[metadata['category'].isin(['healthy', 'CRC', 'adenoma', 'IBD', 'UC', 'CD', 'acute_diarrhoea', 'few_polyps'])]
    # metadata = metadata[metadata['category'].isin(metabolic_conditions + ['healthy'])]
    # metadata = metadata[metadata['category'].isin(cardiovascular_conditions + ['healthy'])]
    # # print(metadata)
    # metadata.reset_index(drop=True, inplace=True)
    # data = data[list(metadata['sample'])]


    # print(metadata)

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        metadata_aux['fold'] = -1
        # print(metadata)
        for fold, (_, val_idx) in enumerate(skf.split(metadata_aux, metadata_aux[args.diagnosis])):
            metadata_aux.loc[val_idx, 'fold'] = fold
        evaluations = []
        indexes = []
        print('---------------------------------------')
        print(condition)
        for fold in metadata_aux['fold'].unique():
            print(fold)


            metadata_train = metadata_aux[metadata_aux['fold'] != fold]
            samples_train = list(metadata_train[args.sample])
            metadata_evaluate = metadata_aux[metadata_aux['fold'] == fold]
            samples_evaluate = list(metadata_evaluate[args.sample])

            # print(samples_fold)
            # print(data.head())
            data_train = data_aux[samples_train]
            data_evaluate = data_aux[samples_evaluate]

            metrics=get_fH_fN_dH_dN(metadata_train,data_train)

            MH,MN=get_MH_MN(metrics,theta_f,theta_d)
            # print(MH)
            # print('.......')
            # print(MN)
            max_len = max(len(MH), len(MN))

            # Pad the shorter list with None or np.nan
            MH_padded = list(MH) + [None] * (max_len - len(MH))
            MN_padded = list(MN) + [None] * (max_len - len(MN))

            # Create the DataFrame
            bacterias = pd.DataFrame({
                'Healthy': MH_padded,
                'Unhealthy': MN_padded
            })
            bacterias.to_csv(f'{condition}_bacterias.csv')
            GMHI=get_all_GMHI(data_evaluate,MH,MN)
            indexes.append(GMHI)
            # print(metadata)
            y_test = []
            for sample in samples_evaluate:
                # print(sample)
                y_test.append(metadata[metadata['sample'] == sample]['category'].iloc[0])
            # GMHI.to_csv('algo.csv')
            # print(GMHI)
            # print(y_test)
            accuracy=get_accuracy2(GMHI,y_test)
            print(accuracy)
            evaluations.append(accuracy)


        GMHI_tax = pd.concat(indexes)
        # GMHI_tax.to_csv(f'{condition}_GMHI_camda2025_preds_taxonomy_corrected_balanced.csv')

    # print(f'_____________MEAN RESULTS___________________')
    # print(mean(t2_results))
    # print(mean(q_results))
    # print(mean(combined_results))

    # results_final2 = pd.concat(results_final, ignore_index=True)
    # print(results_final2)
    # results_final2.to_csv('final_camda2025_pathways_sub3.csv', index = False)

if __name__ == "__main__":
    main()