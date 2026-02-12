# -*- coding: utf-8 -*-
"""
Created on Thu Feb 12 00:01:32 2026

@author: Nishant
"""

# -*- coding: utf-8 -*-
"""
Model Training:
- Linear, Ridge, Lasso
- Decision Tree, Random Forest
- Gradient Boosting, XGBoost
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import xgboost as xgb
from sklearn.pipeline import Pipeline
import joblib

# Load preprocessed data
X_train, X_test, y_train, y_test = pd.read_pickle('data/train_test.pkl')
preprocessor = joblib.load('models/preprocessor.pkl')

# Define models
models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(alpha=1.0),
    'Lasso Regression': Lasso(alpha=0.1),
    'Decision Tree': DecisionTreeRegressor(max_depth=5, random_state=42),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
    'XGBoost': xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42, n_jobs=-1)
}

trained_pipelines = {}

print("\n🤖 Training models...")
for name, model in models.items():
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', model)
    ])
    pipeline.fit(X_train, y_train)
    trained_pipelines[name] = pipeline
    # Save each model
    joblib.dump(pipeline, f'models/{name.replace(" ", "_")}.pkl')
    print(f"✅ {name} trained and saved.")

# Save dictionary of pipelines
joblib.dump(trained_pipelines, 'models/all_models.pkl')
print("\n💾 All models saved in models/all_models.pkl")