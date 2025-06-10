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
    
    # print(selected)
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

    mismatches = ['train_0',
        'train_1',
        'train_1175',
        'train_1177',
        'train_1181',
        'train_1182',
        'train_1183',
        'train_1184',
        'train_1185',
        'train_1186',
        'train_1187',
        'train_14',
        'train_16',
        'train_17',
        'train_172',
        'train_175',
        'train_176',
        'train_180',
        'train_181',
        'train_182',
        'train_183',
        'train_185',
        'train_187',
        'train_1913',
        'train_1914',
        'train_1916',
        'train_1919',
        'train_1920',
        'train_1922',
        'train_1923',
        'train_1924',
        'train_1925',
        'train_1926',
        'train_1927',
        'train_1928',
        'train_1929',
        'train_1932',
        'train_1933',
        'train_1934',
        'train_1936',
        'train_1937',
        'train_1938',
        'train_1939',
        'train_194',
        'train_1940',
        'train_1943',
        'train_1945',
        'train_1949',
        'train_195',
        'train_1950',
        'train_1953',
        'train_1955',
        'train_1958',
        'train_1959',
        'train_1960',
        'train_1962',
        'train_1964',
        'train_1965',
        'train_1967',
        'train_1971',
        'train_1975',
        'train_1976',
        'train_1977',
        'train_1978',
        'train_1979',
        'train_198',
        'train_1981',
        'train_1982',
        'train_1985',
        'train_1987',
        'train_199',
        'train_1990',
        'train_1991',
        'train_1992',
        'train_1993',
        'train_1997',
        'train_200',
        'train_2000',
        'train_2003',
        'train_2006',
        'train_201',
        'train_202',
        'train_203',
        'train_204',
        'train_205',
        'train_206',
        'train_207',
        'train_208',
        'train_209',
        'train_21',
        'train_210',
        'train_211',
        'train_215',
        'train_22',
        'train_221',
        'train_228',
        'train_23',
        'train_255',
        'train_256',
        'train_257',
        'train_258',
        'train_259',
        'train_260',
        'train_261',
        'train_262',
        'train_263',
        'train_264',
        'train_265',
        'train_266',
        'train_267',
        'train_268',
        'train_269',
        'train_270',
        'train_271',
        'train_272',
        'train_273',
        'train_274',
        'train_275',
        'train_276',
        'train_277',
        'train_278',
        'train_2785',
        'train_279',
        'train_2791',
        'train_2794',
        'train_2795',
        'train_2796',
        'train_2799',
        'train_28',
        'train_280',
        'train_2800',
        'train_2802',
        'train_281',
        'train_282',
        'train_283',
        'train_284',
        'train_285',
        'train_286',
        'train_287',
        'train_288',
        'train_289',
        'train_290',
        'train_291',
        'train_292',
        'train_293',
        'train_294',
        'train_295',
        'train_296',
        'train_297',
        'train_298',
        'train_299',
        'train_300',
        'train_3003',
        'train_3004',
        'train_3005',
        'train_3006',
        'train_3007',
        'train_3008',
        'train_301',
        'train_3010',
        'train_3011',
        'train_3012',
        'train_3013',
        'train_3016',
        'train_3017',
        'train_3018',
        'train_3019',
        'train_302',
        'train_3020',
        'train_3021',
        'train_3022',
        'train_3023',
        'train_3024',
        'train_3025',
        'train_3026',
        'train_3027',
        'train_3028',
        'train_3029',
        'train_303',
        'train_3030',
        'train_3031',
        'train_3032',
        'train_3033',
        'train_3034',
        'train_3035',
        'train_3036',
        'train_3037',
        'train_3038',
        'train_3039',
        'train_304',
        'train_3040',
        'train_3041',
        'train_3042',
        'train_3043',
        'train_3044',
        'train_3045',
        'train_3046',
        'train_3047',
        'train_3048',
        'train_3049',
        'train_305',
        'train_3050',
        'train_3051',
        'train_3052',
        'train_3053',
        'train_3054',
        'train_3055',
        'train_3056',
        'train_3058',
        'train_3059',
        'train_306',
        'train_3060',
        'train_3061',
        'train_3062',
        'train_307',
        'train_308',
        'train_309',
        'train_31',
        'train_310',
        'train_311',
        'train_3116',
        'train_3117',
        'train_3118',
        'train_3119',
        'train_312',
        'train_3121',
        'train_3123',
        'train_3126',
        'train_3127',
        'train_313',
        'train_3133',
        'train_3134',
        'train_3135',
        'train_3136',
        'train_3137',
        'train_3138',
        'train_314',
        'train_3141',
        'train_3143',
        'train_3149',
        'train_3150',
        'train_3152',
        'train_3153',
        'train_3158',
        'train_3159',
        'train_316',
        'train_3161',
        'train_3164',
        'train_3165',
        'train_3166',
        'train_317',
        'train_3170',
        'train_3172',
        'train_3173',
        'train_3175',
        'train_3177',
        'train_3178',
        'train_318',
        'train_3181',
        'train_3182',
        'train_3185',
        'train_3188',
        'train_3189',
        'train_319',
        'train_3191',
        'train_320',
        'train_321',
        'train_322',
        'train_323',
        'train_324',
        'train_325',
        'train_326',
        'train_327',
        'train_328',
        'train_329',
        'train_33',
        'train_330',
        'train_3308',
        'train_3309',
        'train_331',
        'train_3310',
        'train_3312',
        'train_3313',
        'train_3314',
        'train_3315',
        'train_3316',
        'train_3317',
        'train_3318',
        'train_3319',
        'train_3321',
        'train_3324',
        'train_3327',
        'train_3330',
        'train_3331',
        'train_3333',
        'train_3335',
        'train_3336',
        'train_3337',
        'train_3338',
        'train_3339',
        'train_3340',
        'train_3344',
        'train_3345',
        'train_3346',
        'train_3347',
        'train_3349',
        'train_3350',
        'train_3351',
        'train_3352',
        'train_3353',
        'train_3354',
        'train_3355',
        'train_3356',
        'train_3357',
        'train_3358',
        'train_3367',
        'train_3368',
        'train_3369',
        'train_3371',
        'train_3372',
        'train_3379',
        'train_3386',
        'train_3387',
        'train_3388',
        'train_3391',
        'train_3392',
        'train_3394',
        'train_3395',
        'train_3398',
        'train_34',
        'train_3403',
        'train_3404',
        'train_3407',
        'train_3409',
        'train_3414',
        'train_3415',
        'train_3416',
        'train_3418',
        'train_3419',
        'train_3420',
        'train_3421',
        'train_3423',
        'train_3424',
        'train_3425',
        'train_3426',
        'train_3429',
        'train_3431',
        'train_3432',
        'train_3436',
        'train_3438',
        'train_3439',
        'train_3440',
        'train_3443',
        'train_3445',
        'train_3446',
        'train_3447',
        'train_3448',
        'train_3449',
        'train_3451',
        'train_3452',
        'train_3454',
        'train_3457',
        'train_3458',
        'train_3459',
        'train_3461',
        'train_3462',
        'train_3463',
        'train_3465',
        'train_3466',
        'train_3467',
        'train_3468',
        'train_3469',
        'train_3472',
        'train_3476',
        'train_3477',
        'train_3479',
        'train_3481',
        'train_3482',
        'train_3484',
        'train_3485',
        'train_3489',
        'train_3490',
        'train_3491',
        'train_3495',
        'train_3496',
        'train_3498',
        'train_3499',
        'train_35',
        'train_3501',
        'train_3504',
        'train_3505',
        'train_3506',
        'train_3510',
        'train_3511',
        'train_3512',
        'train_3514',
        'train_3517',
        'train_3519',
        'train_3520',
        'train_3521',
        'train_3522',
        'train_3523',
        'train_3527',
        'train_3528',
        'train_3530',
        'train_3532',
        'train_3533',
        'train_3534',
        'train_3535',
        'train_3538',
        'train_3539',
        'train_3540',
        'train_3541',
        'train_3542',
        'train_3545',
        'train_3547',
        'train_3548',
        'train_3549',
        'train_3550',
        'train_3552',
        'train_3553',
        'train_3554',
        'train_3557',
        'train_3559',
        'train_3561',
        'train_3564',
        'train_3565',
        'train_3567',
        'train_3568',
        'train_3569',
        'train_3570',
        'train_3572',
        'train_3573',
        'train_3575',
        'train_3576',
        'train_3577',
        'train_3579',
        'train_3580',
        'train_3581',
        'train_3582',
        'train_3583',
        'train_3584',
        'train_3585',
        'train_3587',
        'train_3589',
        'train_3590',
        'train_3592',
        'train_3593',
        'train_3595',
        'train_3596',
        'train_3599',
        'train_36',
        'train_3600',
        'train_3601',
        'train_3602',
        'train_3603',
        'train_3604',
        'train_3606',
        'train_3607',
        'train_3608',
        'train_3609',
        'train_3610',
        'train_3613',
        'train_3614',
        'train_3615',
        'train_3616',
        'train_3617',
        'train_3619',
        'train_3621',
        'train_3622',
        'train_3623',
        'train_3624',
        'train_3626',
        'train_3629',
        'train_3630',
        'train_3632',
        'train_3634',
        'train_3635',
        'train_3637',
        'train_3638',
        'train_3639',
        'train_3640',
        'train_3642',
        'train_3644',
        'train_3645',
        'train_3648',
        'train_3649',
        'train_3650',
        'train_3651',
        'train_3656',
        'train_3657',
        'train_3659',
        'train_3660',
        'train_3661',
        'train_3663',
        'train_3667',
        'train_3668',
        'train_3669',
        'train_3671',
        'train_3672',
        'train_3673',
        'train_3674',
        'train_3675',
        'train_3676',
        'train_3679',
        'train_3680',
        'train_3681',
        'train_3683',
        'train_3684',
        'train_3685',
        'train_3686',
        'train_3687',
        'train_3688',
        'train_3689',
        'train_3690',
        'train_3691',
        'train_3692',
        'train_3693',
        'train_3695',
        'train_3697',
        'train_3698',
        'train_37',
        'train_3700',
        'train_3701',
        'train_3702',
        'train_3703',
        'train_3705',
        'train_3706',
        'train_3708',
        'train_3709',
        'train_3710',
        'train_3716',
        'train_3717',
        'train_3718',
        'train_3719',
        'train_3720',
        'train_3721',
        'train_3723',
        'train_3724',
        'train_3725',
        'train_3726',
        'train_3727',
        'train_3728',
        'train_3731',
        'train_3732',
        'train_3734',
        'train_3736',
        'train_3737',
        'train_3738',
        'train_3739',
        'train_3740',
        'train_3741',
        'train_3744',
        'train_3745',
        'train_3746',
        'train_38',
        'train_3965',
        'train_3967',
        'train_3968',
        'train_3985',
        'train_3986',
        'train_3987',
        'train_3988',
        'train_3989',
        'train_3990',
        'train_3991',
        'train_3992',
        'train_3996',
        'train_4',
        'train_4000',
        'train_4001',
        'train_4002',
        'train_4003',
        'train_4004',
        'train_4005',
        'train_4006',
        'train_4007',
        'train_4008',
        'train_4009',
        'train_4010',
        'train_4011',
        'train_4012',
        'train_4013',
        'train_4014',
        'train_4015',
        'train_4016',
        'train_4017',
        'train_4018',
        'train_4019',
        'train_4020',
        'train_4021',
        'train_4022',
        'train_4023',
        'train_4024',
        'train_4025',
        'train_4026',
        'train_46',
        'train_47',
        'train_472',
        'train_477',
        'train_48',
        'train_49',
        'train_5',
        'train_50',
        'train_51',
        'train_650',
        'train_651',
        'train_652',
        'train_653',
        'train_654',
        'train_655',
        'train_656',
        'train_657',
        'train_658',
        'train_659',
        'train_660',
        'train_661',
        'train_662',
        'train_663',
        'train_664',
        'train_665',
        'train_666',
        'train_667',
        'train_668',
        'train_669',
        'train_670',
        'train_671',
        'train_672',
        'train_673',
        'train_674',
        'train_675',
        'train_676',
        'train_677',
        'train_678',
        'train_679',
        'train_680',
        'train_681',
        'train_682',
        'train_683',
        'train_684',
        'train_685',
        'train_686',
        'train_687',
        'train_688',
        'train_689',
        'train_690',
        'train_691',
        'train_692',
        'train_693',
        'train_694',
        'train_695',
        'train_696',
        'train_697',
        'train_698',
        'train_699',
        'train_700',
        'train_701',
        'train_702',
        'train_703',
        'train_704',
        'train_705',
        'train_706',
        'train_707',
        'train_708',
        'train_709',
        'train_710',
        'train_711',
        'train_712',
        'train_713',
        'train_714',
        'train_715',
        'train_716',
        'train_717',
        'train_718',
        'train_719',
        'train_720',
        'train_721',
        'train_722',
        'train_723',
        'train_724',
        'train_731',
        'train_758',
        'train_795',
        'train_8',
        'train_800',
        'train_801',
        'train_804',
        'train_808',
        'train_812',
        'train_813',
        'train_817',
        'train_819',
        'train_820',
        'train_850',
        'train_852',
        'train_853',
        'train_854',
        'train_855',
        'train_856',
        'train_857',
        'train_858',
        'train_859',
        'train_861',
        'train_862',
        'train_863',
        'train_864',
        'train_865',
        'train_867',
        'train_868',
        'train_871',
        'train_872',
        'train_873',
        'train_874',
        'train_875',
        'train_876',
        'train_877',
        'train_878',
        'train_879',
        'train_880',
        'train_881',
        'train_882',
        'train_883',
        'train_884',
        'train_885',
        'train_886',
        'train_887',
        'train_888',
        'train_889',
        'train_890',
        'train_891',
        'train_892',
        'train_893',
        'train_894',
        'train_895',
        'train_896',
        'train_897',
        'train_898',
        'train_899',
        'train_9',
        'train_900',
        'train_901',
        'train_902',
        'train_903',
        'train_904',
        'train_905',
        'train_906',
        'train_907',
        'train_908',
        'train_909',
        'train_910',
        'train_911',
        'train_912',
        'train_913',
        'train_914',
        'train_915',
        'train_916',
        'train_917',
        'train_918',
        'train_919',
        'train_920',
        'train_921',
        'train_922',
        'train_923',
        'train_924',
        'train_925',
        'train_926',
        'train_927',
        'train_928',
        'train_929',
        'train_930',
        'train_931',
        'train_932',
        'train_933',
        'train_934',
        'train_935',
        'train_936',
        'train_937',
        'train_938',
        'train_939',
        'train_940',
        'train_941',
        'train_942',
        'train_943',
        'train_944',
        'train_945',
        'train_958',
        'train_962',
        'train_980',
        'train_981',
        'train_982',
        'train_983',
        'train_984',
        'train_985',
        'train_986',
        'train_987',
        'train_988',
        'train_989']

    data = data[[item for item in data.columns if item not in mismatches]]
    metadata = metadata[metadata[args.sample].isin(data.columns)]

    #TODO para filtrar los datos
    conditions = dict()

    conditions['1'] = [
        'acute_diarrhoea',
        'STH',
        'respiratoryinf'
    ]

    conditions['2'] = [
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

    conditions['3'] = [
        'hypercholesterolemia',
        'IGT',
        'IGT;MS',
        'metabolic_syndrome',
        'T2D',
        'T2D;respiratoryinf'
    ]

    conditions['4'] = [
        'schizophrenia',
        'ME/CFS',
        'PD'
    ]

    conditions['5'] = [
        'ACVD',
        'CAD',
        'CAD;T2D',
        'HF;CAD',
        'HF;CAD;T2D',
        'hypertension',
        'BD'
    ]

    conditions['6'] = [
        'CD',
        'cirrhosis',
        'IBD',
        'UC'
    ]




    for condition in conditions:


        metadata_aux = metadata[metadata['category'].isin(conditions[condition] + ['healthy'])]
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
        fold_results.to_csv(f'{condition}_f1_score_results_taxonomy_missmatches.csv')

        results_final2 = pd.concat(results_final, ignore_index=True)
        print(results_final2)
        results_final2.to_csv(f'{condition}_camda2025_preds_taxonomy_missmatches.csv', index = False)

    # print(f'_____________MEAN RESULTS___________________')
    # print(mean(t2_results))
    # print(mean(q_results))
    # print(mean(combined_results))

    # results_final2 = pd.concat(results_final, ignore_index=True)
    # print(results_final2)
    # results_final2.to_csv('final_camda2025_pathways_sub3.csv', index = False)

if __name__ == "__main__":
    main()