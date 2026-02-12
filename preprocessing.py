# -*- coding: utf-8 -*-
"""
Created on Thu Feb 12 00:00:43 2026

@author: Nishant
"""

# -*- coding: utf-8 -*-
"""
Preprocessing:
- Split into train/test
- Define numerical & categorical features
- Create ColumnTransformer pipeline
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

def prepare_data(df, target_col='log_charges', test_size=0.2, random_state=42):
    """Split data and return X_train, X_test, y_train, y_test."""
    X = df.drop(['charges', 'log_charges', 'smoker_binary'], axis=1, errors='ignore')
    y = df[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    print(f"📊 Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")
    return X_train, X_test, y_train, y_test, X.columns.tolist()

def get_preprocessor(numerical_features, categorical_features):
    """Create a ColumnTransformer for preprocessing."""
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore', drop='first'))
    ])
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    return preprocessor

if __name__ == "__main__":
    df = pd.read_csv('data/insurance_featured.csv')
    
    # Identify feature types
    X_cols = df.drop(['charges', 'log_charges', 'smoker_binary'], axis=1, errors='ignore').columns
    numerical_features = df[X_cols].select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = df[X_cols].select_dtypes(include=['object', 'category']).columns.tolist()
    
    print("🔢 Numerical features:", numerical_features)
    print("🔠 Categorical features:", categorical_features)
    
    X_train, X_test, y_train, y_test, feature_names = prepare_data(df)
    
    # Save splits
    pd.to_pickle((X_train, X_test, y_train, y_test), 'data/train_test.pkl')
    print("💾 Train/test splits saved to data/train_test.pkl")
    
    # Save preprocessor object
    preprocessor = get_preprocessor(numerical_features, categorical_features)
    joblib.dump(preprocessor, 'models/preprocessor.pkl')
    print("💾 Preprocessor saved to models/preprocessor.pkl")