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
import argparse



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
    with open(f'model_data/{model_name}/scaler.pkl', 'wb') as file:
        pickle.dump(scaler, file)
    selected2 = scaler.transform(selected)
    
    pd.DataFrame(zip(selected.columns, scaler.mean_, scaler.scale_), columns = ['specie', 'mean', 'std']).to_csv(f'model_data/{model_name}/scaling_parameters.csv', index = False)


    # for c in selected.columns:
    #     scaler.fit(np.array(selected[c]).reshape(-1, 1))
    #     selected[c] = scaler.transform(np.array(selected[c]).reshape(-1, 1))
        # params = params.append({'mean':scaler.mean_[0], 'std':scaler.scale_[0]}, ignore_index=True)
        # print(scaler.mean_)
        # print(scaler.scale_)
    # print(selected)
    selected2 = pd.DataFrame(selected2, columns = selected.columns)
    selected2.index = df.T.index

    return selected2


def calculate_pca_stats(df, variance_for_pc = 0.9, alpha = 0.05):
    pca = PCA()

    pca.fit(df)

    with open(f'model_data/{model_name}/pca_model.pkl', 'wb') as file:
        pickle.dump(pca, file)

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
        selected = transform_data(df[[x for x in healthy if x in df.columns]], features)
        
    else:
        if ks:
            features = healthy_features + non_healthy_features
        selected = transform_data(df[[x for x in healthy if x in df.columns]], features)
    
    # print(selected)
    pca, pca_data, D, t2_threshold, C, Q_alpha, fi, threshold_combined = calculate_pca_stats(selected)
    np.save(f'model_data/{model_name}/D_matrix.npy', D)
    np.save(f'model_data/{model_name}/C_matrix.npy', C)
    np.save(f'model_data/{model_name}/fi_matrix.npy', fi)

    thresholds = {'t2':t2_threshold, 'c':Q_alpha, 'combined':threshold_combined}

    with open(f'model_data/{model_name}/thresholds.json', 'w') as json_file:
        json.dump(thresholds, json_file)

    print(t2_threshold, Q_alpha, threshold_combined)

        
    return features, pca, pca_data, D, t2_threshold, C, Q_alpha, fi, threshold_combined, selected

def main():

    '''train with camda samples'''
    global model_name 
    model_name = 'camda_pathways_grouped'
    os.makedirs(f'model_data/{model_name}', exist_ok=True)
    taxonomy = pd.read_csv('new_pathways.csv', index_col = 0)
    metadata = pd.read_csv('../../DataSets/CAMDA/metadata.csv')

    # good_samples = []
    # for c in taxonomy.columns:
    #     if sum(taxonomy[c]) > 90:
    #         good_samples.append(c)
    # taxonomy_aux = taxonomy[good_samples]

    obese = list(metadata[metadata['Diagnosis'] == 'Obese']['SampleID'])
    healthy = list(metadata[metadata['Diagnosis'] == 'Healthy']['SampleID'])
    non_healthy = list(metadata[metadata['Diagnosis'] != 'Healthy']['SampleID'])
    non_healthy = [x for x in non_healthy if x not in obese]

    taxonomy_not_obese = taxonomy[[x for x in taxonomy.columns if x not in obese]]

    features, pca, pca_data, D, t2_threshold, C, Q_alpha, fi, threshold_combined, selected = hiPCA(taxonomy_not_obese, healthy, non_healthy, ks = True, method = 'asymp', only_nonhealthy_features = True)


    # '''train with camda samples'''
    # global model_name 
    # model_name = 'camda_all_samples'
    # os.makedirs(f'model_data/{model_name}', exist_ok=True)
    # taxonomy = pd.read_csv('../../DataSets/CAMDA/taxonomy.txt', sep = '\t', index_col = 0)
    # metadata = pd.read_csv('../../DataSets/CAMDA/metadata.csv')

    # good_samples = []
    # for c in taxonomy.columns:
    #     if sum(taxonomy[c]) > 90:
    #         good_samples.append(c)
    # taxonomy_aux = taxonomy[good_samples]

    # obese = list(metadata[metadata['Diagnosis'] == 'Obese']['SampleID'])
    # healthy = list(metadata[metadata['Diagnosis'] == 'Healthy']['SampleID'])
    # non_healthy = list(metadata[metadata['Diagnosis'] != 'Healthy']['SampleID'])
    # non_healthy = [x for x in non_healthy if x not in obese]

    # taxonomy_not_obese = taxonomy_aux[[x for x in taxonomy_aux.columns if x not in obese]]

    # features, pca, pca_data, D, t2_threshold, C, Q_alpha, fi, threshold_combined, selected = hiPCA(taxonomy_not_obese, healthy, non_healthy, ks = True, method = 'asymp', only_nonhealthy_features = True)

    # '''train with zhu data'''
    # model_name = 'zhu_model'
    # os.makedirs(f'model_data/{model_name}', exist_ok=True)
    # taxonomy = pd.read_csv('data_discovery_zhu_final.csv', index_col = 0)
    # metadata = pd.read_csv('metadata_discovery_zhu.csv', index_col = 0)

    # healthy = list(metadata[metadata['Phenotype'] == 'Healthy'].index)
    # unhealthy = list(metadata[metadata['Phenotype'] != 'Healthy'].index)

    # features, pca, pca_data, D, t2_threshold, C, Q_alpha, fi, threshold_combined, selected = hiPCA(taxonomy, healthy, unhealthy, ks = True, method = 'asymp', only_nonhealthy_features = True)




    # print(selected)



if __name__ == "__main__":
    main()