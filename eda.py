# -*- coding: utf-8 -*-
"""
Created on Wed Feb 11 23:58:03 2026

@author: Nishant
"""

# -*- coding: utf-8 -*-
"""
EDA: statistical summary, missing values, distributions, plots.
Run after data_loader.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from config import FIGURES_DIR

def explore_data(df):
    """Print basic info, statistics, and generate plots."""
    print("\n" + "="*60)
    print("🔍 EXPLORATORY DATA ANALYSIS")
    print("="*60)
    
    print("\n📋 First 5 rows:")
    print(df.head())
    
    print("\n📊 Data Info:")
    print(df.info())
    
    print("\n📈 Statistical Summary:")
    print(df.describe().T)
    
    # Missing values
    missing = df.isnull().sum()
    if missing.sum() == 0:
        print("\n✅ No missing values.")
    else:
        print("\n⚠️ Missing values:")
        print(missing[missing > 0])
    
    # Target variable
    print("\n💰 Target 'charges' statistics:")
    print(f"   Mean:  ${df['charges'].mean():,.2f}")
    print(f"   Median: ${df['charges'].median():,.2f}")
    print(f"   Skew:  {df['charges'].skew():.2f}")
    
    # Categorical columns
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    print(f"\n📌 Categorical features ({len(cat_cols)}): {cat_cols}")
    for col in cat_cols:
        print(f"\n{col}:")
        print(df[col].value_counts())
    
    return df

def plot_eda(df):
    """Create and save EDA plots."""
    print("\n📊 Generating visualizations...")
    
    # 1. Distribution of charges
    fig, axes = plt.subplots(1, 2, figsize=(14,5))
    sns.histplot(df['charges'], kde=True, ax=axes[0], color='steelblue')
    axes[0].set_title('Distribution of Medical Charges')
    sns.boxplot(x=df['charges'], ax=axes[1], color='coral')
    axes[1].set_title('Boxplot of Charges')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'charges_distribution.png'), dpi=150)
    plt.show()
    
    # 2. Charges by smoker
    plt.figure(figsize=(8,6))
    sns.boxplot(x='smoker', y='charges', data=df, palette='Set2')
    plt.title('Medical Charges by Smoking Status')
    plt.savefig(os.path.join(FIGURES_DIR, 'charges_by_smoker.png'), dpi=150)
    plt.show()
    
    # 3. Age vs Charges colored by smoker
    plt.figure(figsize=(10,6))
    sns.scatterplot(x='age', y='charges', hue='smoker', data=df, alpha=0.7)
    plt.title('Age vs Charges (by Smoker)')
    plt.savefig(os.path.join(FIGURES_DIR, 'age_vs_charges.png'), dpi=150)
    plt.show()
    
    # 4. BMI vs Charges colored by smoker
    plt.figure(figsize=(10,6))
    sns.scatterplot(x='bmi', y='charges', hue='smoker', data=df, alpha=0.7)
    plt.title('BMI vs Charges (by Smoker)')
    plt.savefig(os.path.join(FIGURES_DIR, 'bmi_vs_charges.png'), dpi=150)
    plt.show()
    
    # 5. Correlation heatmap (numeric)
    numeric_df = df.select_dtypes(include=[np.number])
    plt.figure(figsize=(8,6))
    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Matrix')
    plt.savefig(os.path.join(FIGURES_DIR, 'correlation_heatmap.png'), dpi=150)
    plt.show()
    
    print(f"✅ Plots saved to {FIGURES_DIR}")

if __name__ == "__main__":
    df = pd.read_csv('data/insurance_raw.csv')
    df = explore_data(df)
    plot_eda(df)