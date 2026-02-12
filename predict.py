# -*- coding: utf-8 -*-
"""
Created on Thu Feb 12 00:02:54 2026

@author: Nishant
"""

# -*- coding: utf-8 -*-
"""
Prediction function for new customers.
Loads best model and preprocessor, applies feature engineering.
"""
import pandas as pd
import numpy as np
import joblib

# Load best model and preprocessor
best_pipeline = joblib.load('models/best_model.pkl')
preprocessor = joblib.load('models/preprocessor.pkl')

def predict_charge(age, sex, bmi, children, smoker, region):
    """
    Predict annual medical insurance charges.
    
    Parameters
    ----------
    age : int
    sex : 'male' or 'female'
    bmi : float
    children : int
    smoker : 'yes' or 'no'
    region : 'southwest', 'southeast', 'northwest', 'northeast'
    
    Returns
    -------
    float : predicted charges in USD
    """
    # Create DataFrame
    input_df = pd.DataFrame({
        'age': [age],
        'sex': [sex],
        'bmi': [bmi],
        'children': [children],
        'smoker': [smoker],
        'region': [region]
    })
    
    # Feature engineering (must match training)
    input_df['bmi_category'] = pd.cut(input_df['bmi'],
                                      bins=[0,18.5,25,30,100],
                                      labels=['Underweight','Normal','Overweight','Obese'])
    input_df['age_group'] = pd.cut(input_df['age'],
                                   bins=[0,30,45,60,100],
                                   labels=['<30','30-45','45-60','60+'])
    input_df['smoker_binary'] = input_df['smoker'].map({'yes':1,'no':0})
    input_df['smoker_bmi'] = input_df['smoker_binary'] * input_df['bmi']
    input_df['smoker_age'] = input_df['smoker_binary'] * input_df['age']
    input_df['has_children'] = (input_df['children'] > 0).astype(int)
    
    # Drop helper columns not used in training
    X_input = input_df.drop(['charges', 'log_charges', 'smoker_binary'], axis=1, errors='ignore')
    
    # Predict
    pred_log = best_pipeline.predict(X_input)[0]
    pred_charge = np.expm1(pred_log)   # back-transform from log
    return pred_charge

# Example usage
if __name__ == "__main__":
    sample = {
        'age': 35,
        'sex': 'male',
        'bmi': 28.5,
        'children': 2,
        'smoker': 'no',
        'region': 'southeast'
    }
    pred = predict_charge(**sample)
    print("\n🔮 Sample Prediction")
    print(f"   Input: {sample}")
    print(f"   Predicted annual charges: ${pred:,.2f}")