# -*- coding: utf-8 -*-
"""
Created on Wed Feb 11 23:59:23 2026

@author: Nishant
"""

# -*- coding: utf-8 -*-
"""
Feature Engineering:
- BMI categories
- Age groups
- Interaction terms
- Log transformation of target
"""
import pandas as pd
import numpy as np

def add_features(df):
    """Add new features to the DataFrame."""
    df_fe = df.copy()
    
    # BMI categories
    df_fe['bmi_category'] = pd.cut(df_fe['bmi'],
                                   bins=[0, 18.5, 25, 30, 100],
                                   labels=['Underweight', 'Normal', 'Overweight', 'Obese'])
    
    # Age groups
    df_fe['age_group'] = pd.cut(df_fe['age'],
                                bins=[0, 30, 45, 60, 100],
                                labels=['<30', '30-45', '45-60', '60+'])
    
    # Interaction: smoker * bmi, smoker * age
    df_fe['smoker_binary'] = df_fe['smoker'].map({'yes': 1, 'no': 0})
    df_fe['smoker_bmi'] = df_fe['smoker_binary'] * df_fe['bmi']
    df_fe['smoker_age'] = df_fe['smoker_binary'] * df_fe['age']
    
    # Has children flag
    df_fe['has_children'] = (df_fe['children'] > 0).astype(int)
    
    # Log transform target (handle skewness)
    df_fe['log_charges'] = np.log1p(df_fe['charges'])   # log(1+charges)
    
    print("✅ Feature engineering completed.")
    print(f"   New columns added: {set(df_fe.columns) - set(df.columns)}")
    return df_fe

if __name__ == "__main__":
    df = pd.read_csv('data/insurance_raw.csv')
    df_fe = add_features(df)
    df_fe.to_csv('data/insurance_featured.csv', index=False)
    print("💾 Featured dataset saved to data/insurance_featured.csv")