# -*- coding: utf-8 -*-
"""
Created on Thu Feb 12 00:02:02 2026

@author: Nishant
"""

# -*- coding: utf-8 -*-
"""
Evaluate all models on test set.
Metrics: R², MAE, RMSE, MAPE (on original scale).
Select best model based on R².
"""
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import joblib

# Load data and models
X_train, X_test, y_train, y_test = pd.read_pickle('data/train_test.pkl')
trained_pipelines = joblib.load('models/all_models.pkl')

# Original charges for back-transformation
df_orig = pd.read_csv('data/insurance_raw.csv')
y_test_charges = df_orig.loc[y_test.index, 'charges'] if hasattr(y_test, 'index') else None

def evaluate_model(pipeline, X_train, X_test, y_train, y_test, y_test_charges=None):
    """Return metrics on both log and original scale."""
    y_pred_train = pipeline.predict(X_train)
    y_pred_test = pipeline.predict(X_test)
    
    # If target was log_charges, back-transform
    if y_test.name == 'log_charges' or y_test_charges is not None:
        y_train_exp = np.expm1(y_train)
        y_test_exp = y_test_charges if y_test_charges is not None else np.expm1(y_test)
        y_pred_train_exp = np.expm1(y_pred_train)
        y_pred_test_exp = np.expm1(y_pred_test)
    else:
        y_train_exp = y_train
        y_test_exp = y_test
        y_pred_train_exp = y_pred_train
        y_pred_test_exp = y_pred_test
    
    return {
        'R2_train': r2_score(y_train_exp, y_pred_train_exp),
        'R2_test': r2_score(y_test_exp, y_pred_test_exp),
        'MAE_test': mean_absolute_error(y_test_exp, y_pred_test_exp),
        'RMSE_test': np.sqrt(mean_squared_error(y_test_exp, y_pred_test_exp)),
        'MAPE_test': np.mean(np.abs((y_test_exp - y_pred_test_exp) / y_test_exp)) * 100
    }

results = {}
for name, pipeline in trained_pipelines.items():
    metrics = evaluate_model(pipeline, X_train, X_test, y_train, y_test, y_test_charges)
    results[name] = metrics
    print(f"\n📌 {name}")
    for k, v in metrics.items():
        if 'MAE' in k or 'RMSE' in k:
            print(f"   {k}: ${v:,.2f}")
        else:
            print(f"   {k}: {v:.4f}")

# Create comparison dataframe
results_df = pd.DataFrame(results).T.sort_values('R2_test', ascending=False)
results_df.to_csv('reports/model_performance.csv')
print("\n🏆 Model Ranking (by R² test):")
print(results_df[['R2_test', 'MAE_test', 'RMSE_test', 'MAPE_test']].round(4))

# Save best model name
best_model_name = results_df.index[0]
with open('models/best_model_name.txt', 'w') as f:
    f.write(best_model_name)
print(f"\n✅ Best model: {best_model_name}")