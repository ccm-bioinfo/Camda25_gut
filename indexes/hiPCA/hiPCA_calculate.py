import pandas as pd
import pickle
import joblib
import numpy as np
import json
import argparse


def custom_transform(x):
    if x <= 1:
        return np.log2(2 * x + 0.00001)
    else:
        return np.sqrt(x)

def transform_data(df):
    scaling_data = pd.read_csv(f'{path}/scaling_parameters.csv')
    features = list(scaling_data['specie'])
    # print(df.index)
    with open(f'{path}/scaler.pkl', 'rb') as file:
        scaler = pickle.load(file)
    # print(features)
    # scaler = StandardScaler()
    aux = pd.DataFrame()
    for item in list(set(features)):
        if item in df.index:
            aux[item] = list(df.T[item])
        else:
            aux[item] = [0 for x in range(len(df.T))]
    selected = aux.applymap(custom_transform)

    # scaler.fit(np.array(selected))
    selected = selected[features]
    selected2 = scaler.transform(selected)
    selected2 = pd.DataFrame(selected2, columns = selected.columns)
    selected2.index = df.T.index
    

    return selected2

def calculate_index(data_transformed):
    D = np.load(f'{path}/D_matrix.npy')
    C = np.load(f'{path}/C_matrix.npy')
    fi = np.load(f'{path}/fi_matrix.npy')
    pca = joblib.load(f'{path}/pca_model.pkl')


    with open(f'{path}/thresholds.json', 'r') as file:
        thresholds = json.load(file)
        t2_threshold = thresholds['t2']
        Q_alpha = thresholds['c']
        threshold_combined = thresholds['combined']

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

def calculate_hiPCA(path_, data):
    global path
    path = path_
    data = transform_data(data)
    results = calculate_index(data)
    return results

def main():
    # parser.add_argument("--name", required=True, type=str, help="Name of the model")
    parser = argparse.ArgumentParser(description="A script that requires --name and --path arguments.")
    parser.add_argument("--path", required=True, type=str, help="Path to the model directory") 
    parser.add_argument("--input", required=True, type=str, help="Path to the input file")
    parser.add_argument("--outdir", required=True, type=str, help="Path to the output directory")
    args = parser.parse_args()

    global path
    path = args.path

    # taxonomy = pd.read_csv('../../DataSets/CAMDA/taxonomy.txt', sep = '\t', index_col = 0)
    # taxonomy = pd.read_csv('../../DataSets/COVID/CAMDA_taxa.txt', sep = '\t', index_col = 0)

    data = pd.read_csv(args.input, sep = '\t', index_col = 0)
    data = transform_data(data)
    # print(data)
    # data.to_csv('transformed.csv')
    results = calculate_index(data)
    results.to_csv(args.outdir.replace('/', '') +'/hiPCA_results.csv')




if __name__ == "__main__":
    main()