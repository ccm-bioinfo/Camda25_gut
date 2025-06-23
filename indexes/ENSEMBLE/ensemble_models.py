# import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score
import numpy as np
import pandas as pd

condition = 'DD'
hipca_preds = pd.read_csv(f'../hiPCA/{condition}_camda2025_preds_taxonomy_corrected_balanced.csv', index_col = 0)
gmhi_preds = pd.read_csv(f'../RF_GMHI/{condition}_GMHI_camda2025_preds_taxonomy_corrected_balanced.csv', index_col = 0)
metadata = pd.read_csv('../../DataSets/CAMDA_2025/metadata_corrected_final.txt', sep = '\t')
pathways = pd.read_csv('../../DataSets/Path_agrupados.txt', sep = '\t', index_col = 0)
# print(pathways.columns)
pathways.drop('NA', inplace = True, axis = 1)

# Join the prediction DataFrames on the index (sample IDs)
X = hipca_preds.join(gmhi_preds, how='inner').join(pathways, how='inner')  # Inner join ensures only overlapping samples are used
# print(X.columns)
X.drop('Group', inplace = True, axis = 1)
# Merge with metadata to get the target labels
metadata = metadata.set_index('sample')
y = metadata.loc[X.index, 'category']

# Optional: check for NaNs
assert not X.isnull().any().any(), "Missing values in X"
assert not y.isnull().any(), "Missing labels in y"


X.drop('Prediction T2', inplace = True, axis = 1)
X.drop('Prediction Q', inplace = True, axis = 1)
X.drop('Combined Prediction', inplace = True, axis = 1)
# X.drop('Predicition T2', inplace = True, axis = 1)
# print(X.columns)
# print(list(y))


metadata_aux = metadata.loc[X.index].copy()
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
metadata_aux['fold'] = -1
for fold, (_, test_index) in enumerate(skf.split(X, y)):
    metadata_aux.iloc[test_index, metadata_aux.columns.get_loc('fold')] = fold

# Evaluate decision tree in each fold
f1_results = []
for fold in metadata_aux['fold'].unique():
    print(f"Fold {fold}")
    
    metadata_train = metadata_aux[metadata_aux['fold'] != fold]
    samples_train = metadata_train.index
    metadata_eval = metadata_aux[metadata_aux['fold'] == fold]
    samples_eval = metadata_eval.index
    
    X_train = X.loc[samples_train]
    y_train = y.loc[samples_train]
    X_eval = X.loc[samples_eval]
    y_eval = y.loc[samples_eval]

    # X_train.to_csv('algo.csv')
    # Train a decision tree
    clf = DecisionTreeClassifier(max_depth=5, random_state=42)
    clf.fit(X_train, y_train)

    # Predict
    y_pred = clf.predict(X_eval)
    
    # Convert labels to 0/1 for f1_score (assuming 'healthy' = 1, other = 0)
    y_eval_bin = [1 if x == 'healthy' else 0 for x in y_eval]
    y_pred_bin = [1 if x == 'healthy' else 0 for x in y_pred]
    
    f1 = f1_score(y_eval_bin, y_pred_bin, pos_label=0)
    print(f"F1 score (non-healthy as positive class): {f1:.4f}")
    f1_results.append(f1)

# Report all fold results
fold_results = pd.DataFrame(f1_results, columns=['F1']).T
fold_results.columns = [f'Fold{i+1}' for i in range(len(f1_results))]
fold_results['Mean'] = fold_results.mean(axis=1)
print("\nFold-wise F1 scores and Mean:")
print(fold_results)
fold_results.to_csv(f'{condition}_ensemble_results.tsv')