from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np

# Load data
condition = 'ALL'
hipca_preds = pd.read_csv(f'../hiPCA/output/{condition}_camda2025_preds_taxonomy_corrected_balanced_final.csv', index_col=0)
gmhi_preds = pd.read_csv(f'../RF_GMHI/output/{condition}_GMHI_camda2025_preds_taxonomy_corrected_balanced.csv', index_col=0)
metadata = pd.read_csv('../../DataSets/CAMDA_2025/metadata_corrected_final.txt', sep='\t')
pathways = pd.read_csv('../../DataSets/Path_agrupados.txt', sep='\t', index_col=0)
pathways.drop('NA', inplace=True, axis=1)

# Prepare features
X = hipca_preds.join(gmhi_preds, how='inner').join(pathways, how='inner')
X.drop(['Group', 'Prediction T2', 'Prediction Q', 'Combined Prediction'], axis=1, inplace=True)

# Prepare binary labels
metadata = metadata.set_index('sample')
y = metadata.loc[X.index, 'category']
y_binary = y.apply(lambda x: 'healthy' if x == 'healthy' else 'unhealthy')  # Only two classes

assert not X.isnull().any().any(), "Missing values in X"
assert not y_binary.isnull().any(), "Missing labels in y"

# Prepare Stratified K-Folds
metadata_aux = metadata.loc[X.index].copy()
metadata_aux['fold'] = -1
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for fold, (_, test_index) in enumerate(skf.split(X, y_binary)):
    metadata_aux.iloc[test_index, metadata_aux.columns.get_loc('fold')] = fold

# Prepare prediction storage
results_df = pd.DataFrame(index=X.index, columns=['true_label', 'predicted_label', 'proba_healthy'])

# For storing feature importances
all_feature_importances = pd.DataFrame()

# Cross-validation loop
f1_results = []
for fold in range(5):
    print(f"Fold {fold}")

    train_idx = metadata_aux[metadata_aux['fold'] != fold].index
    test_idx = metadata_aux[metadata_aux['fold'] == fold].index

    X_train, y_train = X.loc[train_idx], y_binary.loc[train_idx]
    X_eval, y_eval = X.loc[test_idx], y_binary.loc[test_idx]

    # Define pipeline: StandardScaler + LogisticRegression
    clf = Pipeline([
        ('scaler', StandardScaler()),
        ('logreg', LogisticRegression(solver='liblinear', random_state=42))
    ])

    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_eval)
    y_proba = clf.predict_proba(X_eval)

    # Extract probability of 'healthy' class
    class_labels = clf.named_steps['logreg'].classes_
    healthy_index = np.where(class_labels == 'healthy')[0][0]
    healthy_proba = y_proba[:, healthy_index]

    results_df.loc[test_idx, 'true_label'] = y_eval
    results_df.loc[test_idx, 'predicted_label'] = y_pred
    results_df.loc[test_idx, 'proba_healthy'] = healthy_proba

    # Binary F1 score (with unhealthy = 1 as positive class)
    y_eval_bin = (y_eval == 'unhealthy').astype(int)
    y_pred_bin = (y_pred == 'unhealthy').astype(int)
    f1 = f1_score(y_eval_bin, y_pred_bin, pos_label=1)
    print(f"F1 score (unhealthy as positive class): {f1:.4f}")
    f1_results.append(f1)

    # Extract feature importances (logistic regression coefficients)
    coefs = clf.named_steps['logreg'].coef_[0]
    feature_names = X.columns
    feature_importance_df = pd.DataFrame({
        'feature': feature_names,
        'coefficient': coefs,
        'fold': fold
    })
    all_feature_importances = pd.concat([all_feature_importances, feature_importance_df], ignore_index=True)

# Confusion matrix
results_df.dropna(inplace=True)
true_bin = (results_df['true_label'] == 'unhealthy').astype(int)
pred_bin = (results_df['predicted_label'] == 'unhealthy').astype(int)
conf_matrix = confusion_matrix(true_bin, pred_bin)

# Display and save results
fold_results = pd.DataFrame(f1_results, columns=['F1']).T
fold_results.columns = [f'Fold{i+1}' for i in range(len(f1_results))]
fold_results['Mean'] = fold_results.mean(axis=1)

print("\nFold-wise F1 scores and Mean:")
print(fold_results)

print("\nBinary Confusion Matrix (Healthy=0, Unhealthy=1):")
print(conf_matrix)

# Save outputs
results_df.to_csv(f'output/{condition}_binary_ensemble_predictions_logreg_final.tsv', sep='\t')
fold_results.to_csv(f'output/{condition}_binary_ensemble_f1_scores_logreg_final.tsv', sep='\t')
pd.DataFrame(conf_matrix, index=['true_healthy', 'true_unhealthy'], columns=['pred_healthy', 'pred_unhealthy']).to_csv(f'output/{condition}_binary_confusion_matrix_logreg_final.tsv', sep='\t')

# Save feature importance
all_feature_importances.to_csv(f'output/{condition}_logreg_feature_importances_by_fold.tsv', sep='\t', index=False)
