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


def ks_test(df, healthy, non_healthy, method_ks = 'asymp', p_val = 0.001):
    healthy_df = df[[x for x in df.columns if x in healthy]].T
    nonhealthy_df = df[[x for x in df.columns if x in non_healthy]].T
    healthy_features = []
    nonhealthy_features = []
    for feature in list(df.index):
        if kstest(list(healthy_df[feature]), list(nonhealthy_df[feature]), alternative = 'less', method = method_ks).pvalue <= p_val:
            healthy_features.append(feature)
        if kstest(list(nonhealthy_df[feature]), list(healthy_df[feature]), alternative = 'less', method = method_ks).pvalue <= p_val:
            nonhealthy_features.append(feature)
    print(f'# Healthy features selected by KS: {len(healthy_features)}')
    print(f'# Unheatlhy features selected by KS: {len(nonhealthy_features)}')
    return healthy_features, nonhealthy_features

def custom_transform(x):
    if x <= 1:
        return np.log2(2 * x + 0.00001)
    else:
        return np.sqrt(x)

def transform_data(df, features):
    scaler = StandardScaler()
    aux = pd.DataFrame()
    for item in list(set(features)):
        if item in df.index:
            aux[item] = list(df.T[item])
        else:
            aux[item] = [0 for x in range(len(df.T))]
    selected = aux.applymap(custom_transform)

    scaler.fit(selected)
    # with open(f'model_data/{model_name}/scaler.pkl', 'wb') as file:
    #     pickle.dump(scaler, file)
    selected2 = scaler.transform(selected)

    scaling_data = pd.DataFrame(zip(selected.columns, scaler.mean_, scaler.scale_), columns = ['specie', 'mean', 'std'])
    
    # pd.DataFrame(zip(selected.columns, scaler.mean_, scaler.scale_), columns = ['specie', 'mean', 'std']).to_csv(f'model_data/{model_name}/scaling_parameters.csv', index = False)


    # for c in selected.columns:
    #     scaler.fit(np.array(selected[c]).reshape(-1, 1))
    #     selected[c] = scaler.transform(np.array(selected[c]).reshape(-1, 1))
        # params = params.append({'mean':scaler.mean_[0], 'std':scaler.scale_[0]}, ignore_index=True)
        # print(scaler.mean_)
        # print(scaler.scale_)
    # print(selected)
    selected2 = pd.DataFrame(selected2, columns = selected.columns)
    selected2.index = df.T.index

    return selected2, scaling_data, scaler


def calculate_pca_stats(df, variance_for_pc = 0.9, alpha = 0.05):
    pca = PCA()

    pca.fit(df)

    # with open(f'model_data/{model_name}/pca_model.pkl', 'wb') as file:
    #     pickle.dump(pca, file)

    eigenvalues = pca.explained_variance_
    eigenvectors = pca.components_
    singular = pca.singular_values_
    
    pca_data = pd.DataFrame(zip(eigenvectors, eigenvalues, singular), columns = ('Eigenvectors', 'Explained_variance', 'Singular_values')).sort_values('Explained_variance', ascending = False)
    pca_data['%variance'] = pca_data['Explained_variance'] / sum(pca_data['Explained_variance'])
    pca_data = pca_data.sort_values('%variance', ascending = False)
    pca_data['%variance_cumulative'] = pca_data['%variance'].cumsum()
    
    principal_components = list(pca_data[pca_data['%variance_cumulative'] < variance_for_pc]['Eigenvectors'])
    print(f'# Principal Components selected: {len(principal_components)}')
    
    principal_values = list(pca_data[pca_data['%variance_cumulative'] < variance_for_pc]['Explained_variance'])
    D = np.array(principal_components).T @ np.linalg.inv(np.diag(principal_values)) @ np.array(principal_components)
    deg_free = len(principal_components) 
    # alpha = 0.05
    t2_threshold = chi2.ppf(1-alpha, deg_free)
#     print(1-alpha, deg_free)
#     print(t2_threshold)
    
    principal_components_residual = list(pca_data[pca_data['%variance_cumulative'] >= variance_for_pc]['Eigenvectors'])
    principal_values_residual = list(pca_data[pca_data['%variance_cumulative'] >= variance_for_pc]['Explained_variance'])
    principal_singvalues_residual = list(pca_data[pca_data['%variance_cumulative'] >= variance_for_pc]['Singular_values'])
    
    C = np.array(principal_components_residual).T @ np.array(principal_components_residual)
    Theta1 = sum(principal_values_residual)
    Theta2 = sum([x**2 for x in principal_values_residual])
    Theta3 = sum([x**3 for x in principal_values_residual])
    
    c_alpha = norm.ppf(1-alpha)
    
    h0 = 1-((2*Theta1*Theta3)/(3*(Theta2**2)))

    '''INTENTO'''
    Q_alpha = Theta1*(((((c_alpha*np.sqrt(2*Theta2*(h0**2)))/Theta1)+1+((Theta2*h0*(h0-1))/(Theta1**2))))**(1/h0))
    # print(Q_alpha)
    Q_alpha = Theta1*(((((np.sqrt(c_alpha*(2*Theta2*(h0**2))))/Theta1)+1-((Theta2*h0*(h0-1))/(Theta1**2))))**(1/h0))
    # print(Q_alpha)
    Q_alpha = (Theta2/Theta1) * chi2.ppf(alpha, len(principal_components_residual)) * ((Theta1**2)/Theta2)
    # print(Q_alpha)
    
    # Q_alpha = Theta1*(((((c_alpha*np.sqrt(2*Theta2*(h0**2)))/Theta1)+1+((Theta2*h0*(h0-1))/(Theta1**2))))**(1/h0))
    
    #fi = D/t2_threshold + (np.eye(len(principal_components[0])) - (np.array(principal_components).T @ np.array(principal_components)))/Q_alpha
    fi = D/t2_threshold  + C/Q_alpha
    g = ((len(principal_components) / t2_threshold**2) + (Theta2 / Q_alpha**2)) / ((len(principal_components)/t2_threshold) + (Theta1 / Q_alpha))
    h = ((len(principal_components)/t2_threshold) + (Theta1 / Q_alpha))**2 / ((len(principal_components) / t2_threshold**2) + (Theta2 / Q_alpha**2))

    chi_value = chi2.ppf(1-alpha, h)
    threshold_combined = g*chi_value
    
    return pca, pca_data, D, t2_threshold, C, Q_alpha, fi, threshold_combined

def hotelling_t2(df, pca, pca_data, variance_for_pc = 0.9, alpha = 0.05):
    principal_components = list(pca_data[pca_data['%variance_cumulative'] < variance_for_pc]['Eigenvectors'])
    print(f'# Principal Components selected: {len(principal_components)}')
    principal_values = list(pca_data[pca_data['%variance_cumulative'] < variance_for_pc]['Explained_variance'])
    D = np.array(principal_components).T @ np.linalg.inv(np.diag(principal_values)) @ np.array(principal_components)
    deg_free = len(principal_components) 
    # alpha = 0.05
    t2_threshold = chi2.ppf(1-alpha, deg_free)
    T2 = []
    pred = []
    
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
            
    hoteling = pd.DataFrame(zip(df.index, T2, pred), columns = ['Sample', 'T2', 'Prediction T2'])
    
    return D, principal_components, hoteling, t2_threshold

def Q_statistic(df, pca, pca_data, variance_for_pc = 0.9, alpha = 0.05):
    principal_components_residual = list(pca_data[pca_data['%variance_cumulative'] >= variance_for_pc]['Eigenvectors'])
    principal_values_residual = list(pca_data[pca_data['%variance_cumulative'] >= variance_for_pc]['Explained_variance'])
    principal_singvalues_residual = list(pca_data[pca_data['%variance_cumulative'] >= variance_for_pc]['Singular_values'])
    
    C = np.array(principal_components_residual).T @ np.array(principal_components_residual)
    Theta1 = sum(principal_values_residual)
    Theta2 = sum([x**2 for x in principal_values_residual])
    Theta3 = sum([x**3 for x in principal_values_residual])
    
    c_alpha = norm.ppf(1-alpha)
    
    h0 = 1-((2*Theta1*Theta3)/(3*Theta2**2))

    #! NO BORRAR ORIGINAL
    Q_alpha = Theta1*(((((c_alpha*np.sqrt(2*Theta2*(h0**2)))/Theta1)+1+((Theta2*h0*(h0-1))/(Theta1**2))))**(1/h0))
    print(Q_alpha)
    Q_alpha = Theta1*(((((np.sqrt(c_alpha*(2*Theta2*(h0**2))))/Theta1)+1-((Theta2*h0*(h0-1))/(Theta1**2))))**(1/h0))
    print(Q_alpha)
    Q_alpha = (Theta2/Theta1) * chi2.ppf(1-alpha, len(principal_components_residual)) * ((Theta1**2)/Theta2)
    print(Q_alpha)
    
    Q = []
    pred = []
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
    
    Q_statistic = pd.DataFrame(zip(df.index, Q, pred), columns = ['Sample', 'Q', 'Prediction Q'])
    
    return C, Theta1, Theta2, Q_statistic, Q_alpha
    

def combined_index(df, D, t2_threshold, principal_components, Q_alpha, Theta1, Theta2, pca, alpha = 0.05):
    fi = D/t2_threshold + (np.eye(len(principal_components[0])) - (np.array(principal_components).T @ np.array(principal_components)))/Q_alpha
    g = ((len(principal_components) / t2_threshold**2) + (Theta2 / Q_alpha**2)) / ((len(principal_components)/t2_threshold) + (Theta1 / Q_alpha))
    h = ((len(principal_components)/t2_threshold) + (Theta1 / Q_alpha))**2 / ((len(principal_components) / t2_threshold**2) + (Theta2 / Q_alpha**2))

    chi_value = chi2.ppf(1-alpha, h)
    threshold_combined = g*chi_value
    combined = []
    pred = []

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

    combined = pd.DataFrame(zip(df.index, combined, pred), columns = ['Sample', 'Combined', 'Prediction Combined']) 
    return combined

def hiPCA(df, healthy, non_healthy, features = [], ks = False, method = 'auto', p_val = 0.001, only_nonhealthy_features = False):
    if ks:
        healthy_features, non_healthy_features = ks_test(df, healthy, non_healthy, method_ks = method, p_val = p_val)
        
    if only_nonhealthy_features:
        healthy_features = []
        if ks:
            features = healthy_features + non_healthy_features
        selected, scaling_data, scaler = transform_data(df[[x for x in healthy if x in df.columns]], features)
   
    else:
        if ks:
            features = healthy_features + non_healthy_features
        selected, scaling_data, scaler = transform_data(df[[x for x in healthy if x in df.columns]], features)
    
    # scaling_data.to_csv(f'{condition}_bacterias.csv')
    # print(scaling_data)
    # print(selected)
    # p max_len = max(healthy_features, non_healthy_features)

    # # Pad the shorter list with None or np.nan
    # MH_padded = list(MH) + [None] * (max_len - len(MH))
    # MN_padded = list(MN) + [None] * (max_len - len(MN))

    # # Create the DataFrame
    # bacterias = pd.DataFrame({
    #             'Healthy': MH_padded,
    #             'Unhealthy': MN_padded
    # })
    # bacterias.to_csv(f'{condition}_bacterias.csv')

    pca, pca_data, D, t2_threshold, C, Q_alpha, fi, threshold_combined = calculate_pca_stats(selected)
    # np.save(f'model_data/{model_name}/D_matrix.npy', D)
    # np.save(f'model_data/{model_name}/C_matrix.npy', C)
    # np.save(f'model_data/{model_name}/fi_matrix.npy', fi)

    thresholds = {'t2':t2_threshold, 'c':Q_alpha, 'combined':threshold_combined}
    

    # with open(f'model_data/{model_name}/thresholds.json', 'w') as json_file:
    #     json.dump(thresholds, json_file)

    print(t2_threshold, Q_alpha, threshold_combined)

        
    return features, pca, pca_data, D, t2_threshold, C, Q_alpha, fi, threshold_combined, selected, scaling_data, scaler

def transform_data_evaluate(df, scaling_data, scaler):
    # scaling_data = pd.read_csv(f'{path}/scaling_parameters.csv')
    features = list(scaling_data['specie'])
    # print(df.index)
    # with open(f'{path}/scaler.pkl', 'rb') as file:
    #     scaler = pickle.load(file)
    # print(features)
    # scaler = StandardScaler()
    aux = pd.DataFrame()
    for item in list(set(features)):
        # print(item)
        if item in df.index:
            aux[item] = list(df.T[item])
        else:
            aux[item] = [0 for x in range(len(df.T))]
    # print(aux)
    selected = aux.applymap(custom_transform)

    # scaler.fit(np.array(selected))
    # print(selected)
    selected = selected[features]
    # print(selected)
    selected2 = scaler.transform(selected)
    selected2 = pd.DataFrame(selected2, columns = selected.columns)
    selected2.index = df.T.index
    

    return selected2

def calculate_index(data_transformed, C, D, fi, pca, t2_threshold, Q_alpha, threshold_combined):
    # D = np.load(f'{path}/D_matrix.npy')
    # C = np.load(f'{path}/C_matrix.npy')
    # fi = np.load(f'{path}/fi_matrix.npy')
    # pca = joblib.load(f'{path}/pca_model.pkl')


    # with open(f'{path}/thresholds.json', 'r') as file:
    #     thresholds = json.load(file)
    #     t2_threshold = thresholds['t2']
    #     Q_alpha = thresholds['c']
    #     threshold_combined = thresholds['combined']

    T2, Q, combined = [], [], []
    pred_t2, pred_Q, pred_combined = [], [], []

    try:
        for item in pca.transform(data_transformed):
            index = item.T @ D @ item
            index2 = item.T @ C @ item
            index3 = item.T @ fi @ item
            T2.append(index)
            Q.append(index2)
            combined.append(index3)
            if index > t2_threshold:
                pred_t2.append('Unhealthy')
            else:
                pred_t2.append('Healthy')

            if index2 > Q_alpha:
                pred_Q.append('Unhealthy')
            else:
                pred_Q.append('Healthy')

            if index3 > threshold_combined:
                pred_combined.append('Unhealthy')
            else:
                pred_combined.append('Healthy') 
    except:
        for item in np.array(data_transformed):
            index = item.T @ D @ item
            index2 = item.T @ C @ item
            index3 = item.T @ fi @ item
            T2.append(index)
            Q.append(index2)
            combined.append(index3)
            if index > t2_threshold:
                pred_t2.append('Unhealthy')
            else:
                pred_t2.append('Healthy')

            if index2 > Q_alpha:
                pred_Q.append('Unhealthy')
            else:
                pred_Q.append('Healthy')

            if index3 > threshold_combined:
                pred_combined.append('Unhealthy')
            else:
                pred_combined.append('Healthy') 

    return pd.DataFrame(zip(data_transformed.index, T2, pred_t2, Q, pred_Q, combined, pred_combined), columns = ['SampleID', 'T2', 'Prediction T2', 'Q', 'Prediction Q', 'Combined Index', 'Combined Prediction'])

def calculate_hiPCA(path_, data, C, D, fi, pca, t2_threshold, Q_alpha, threshold_combined):
    global path
    path = path_
    data = transform_data(data)
    results = calculate_index(data)
    return results

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
        t2_results = []
        q_results = []
        combined_results = []
        results_final = []
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

            # print(metadata_fold.head())
            # print(data_fold.head())

            healthy = list(metadata_train[metadata_train[args.diagnosis] == args.control][args.sample])
            non_healthy = list(metadata_train[metadata_train[args.diagnosis] != args.control][args.sample])


            features, pca, pca_data, D, t2_threshold, C, Q_alpha, fi, threshold_combined, selected, scaling, scaler = hiPCA(data_train, healthy, non_healthy, ks = True, method = args.method, only_nonhealthy_features = True)
            scaling.to_csv(f'{condition}_bacteria.csv')
            data_evaluate = transform_data_evaluate(data_evaluate, scaling, scaler)
            # print(data)
            # data.to_csv('transformed.csv')
            results = calculate_index(data_evaluate, C, D, fi, pca, t2_threshold, Q_alpha, threshold_combined)
            # print(results['Prediction T2'])
            results['T2'] = results['T2'] - t2_threshold
            results['Q'] = results['Q'] - Q_alpha
            results['Combined Index'] = results['Combined Index'] - threshold_combined
            # print(results)
            results_final.append(results)

            t2_preds = [1 if x == 'Healthy' else 0 for x in results['Prediction T2']]
            q_preds =  [1 if x == 'Healthy' else 0 for x in results['Prediction Q']]
            combined_preds =  [1 if x == 'Healthy' else 0 for x in results['Combined Prediction']]

            true_labels = [1 if x == 'healthy' else 0 for x in metadata_evaluate[args.diagnosis]]
            # print(t2_preds)
            t2_result = f1_score(true_labels, t2_preds, pos_label = 0)
            q_result = f1_score(true_labels, q_preds, pos_label = 0)
            combined_result = f1_score(true_labels, combined_preds, pos_label = 0)

            t2_results.append(t2_result)
            q_results.append(q_result)
            combined_results.append(combined_result)

            print(t2_result)
            print(q_result)
            print(combined_result)

        fold_results = pd.DataFrame(zip(t2_results, q_results, combined_results), columns = ['T2', 'Q', 'Combined'])
        fold_results = fold_results.T
        fold_results.columns = ['Fold1', 'Fold2', 'Fold3', 'Fold4', 'Fold5']
        fold_results['Mean'] = fold_results.mean(axis=1)
        fold_results.to_csv(f'{condition}_f1_score_results_taxonomy_corrected_balanced.csv')

        results_final2 = pd.concat(results_final, ignore_index=True)
        print(results_final2)
        results_final2.to_csv(f'{condition}_camda2025_preds_taxonomy_corrected_balanced.csv', index = False)

    # print(f'_____________MEAN RESULTS___________________')
    # print(mean(t2_results))
    # print(mean(q_results))
    # print(mean(combined_results))

    # results_final2 = pd.concat(results_final, ignore_index=True)
    # print(results_final2)
    # results_final2.to_csv('final_camda2025_pathways_sub3.csv', index = False)

if __name__ == "__main__":
    main()