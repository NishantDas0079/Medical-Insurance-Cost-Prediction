# -*- coding: utf-8 -*-
"""
Created on Wed Feb 11 23:53:30 2026

@author: Nishant
"""

# -*- coding: utf-8 -*-
"""
Configuration file – shared settings and imports.
Run this first to set up environment.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ==================== PATHS ====================
ZIP_PATH = r"C:\Users\Nishant\Downloads\archive.zip"
DATA_DIR = "data"
MODELS_DIR = "models"
REPORTS_DIR = "reports"
FIGURES_DIR = os.path.join(REPORTS_DIR, "figures")

# Create directories
for d in [DATA_DIR, MODELS_DIR, REPORTS_DIR, FIGURES_DIR]:
    os.makedirs(d, exist_ok=True)

# ==================== PLOT STYLE ====================
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('husl')
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

print("✅ Config loaded – directories ready.")

# -*- coding: utf-8 -*-
"""
Created on Wed Feb 11 23:54:32 2026

@author: Nishant
"""

# -*- coding: utf-8 -*-
"""
Load insurance dataset from ZIP file or download fallback.
Run after 00_config.py
"""
import zipfile
import pandas as pd
from config import ZIP_PATH, DATA_DIR

def load_insurance_data():
    """Extract insurance.csv from archive.zip and load into DataFrame."""
    if os.path.exists(ZIP_PATH):
        print(f"📦 Found archive: {ZIP_PATH}")
        with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
            # Find the CSV file (case-insensitive)
            csv_files = [f for f in zip_ref.namelist() 
                         if 'insurance' in f.lower() and f.endswith('.csv')]
            if not csv_files:
                # Take any CSV if insurance not found
                csv_files = [f for f in zip_ref.namelist() if f.endswith('.csv')]
            csv_file = csv_files[0]
            print(f"📄 Extracting: {csv_file}")
            zip_ref.extract(csv_file, DATA_DIR)
            csv_path = os.path.join(DATA_DIR, os.path.basename(csv_file))
            df = pd.read_csv(csv_path)
            print(f"✅ Loaded: {csv_path}")
            print(f"   Shape: {df.shape[0]} rows × {df.shape[1]} columns")
            return df
    else:
        print(f"❌ Archive not found at {ZIP_PATH}")
        print("📥 Downloading fallback from GitHub...")
        url = "https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv"
        df = pd.read_csv(url)
        df.to_csv(os.path.join(DATA_DIR, 'insurance.csv'), index=False)
        print("✅ Downloaded and saved.")
        return df

if __name__ == "__main__":
    df = load_insurance_data()
    # Save raw data for later use
    df.to_csv(os.path.join(DATA_DIR, 'insurance_raw.csv'), index=False)
    print("💾 Raw data saved to data/insurance_raw.csv")
    
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

# -*- coding: utf-8 -*-
"""
Created on Thu Feb 12 00:02:53 2026

@author: Nishant
"""

# -*- coding: utf-8 -*-
"""
Business insights from the data and model.
"""
import pandas as pd
import numpy as np
import joblib

df = pd.read_csv('data/insurance_raw.csv')
best_pipeline = joblib.load('models/best_model.pkl')  # from 06_evaluation

print("\n" + "="*60)
print("💡 BUSINESS INSIGHTS & RECOMMENDATIONS")
print("="*60)

# 1. Smoker impact
smoker_cost = df.groupby('smoker')['charges'].mean()
premium_increase = (smoker_cost['yes'] - smoker_cost['no']) / smoker_cost['no'] * 100
print(f"\n1️⃣ Smokers pay {premium_increase:.1f}% higher premiums on average.")
print(f"   Non-smoker: ${smoker_cost['no']:,.0f} | Smoker: ${smoker_cost['yes']:,.0f}")

# 2. Obesity impact
obese_cost = df[df['bmi'] >= 30]['charges'].mean()
normal_cost = df[(df['bmi'] >= 18.5) & (df['bmi'] < 25)]['charges'].mean()
obese_impact = (obese_cost - normal_cost) / normal_cost * 100
print(f"\n2️⃣ Obese customers (BMI≥30) have {obese_impact:.1f}% higher charges than normal weight.")
print(f"   Normal weight: ${normal_cost:,.0f} | Obese: ${obese_cost:,.0f}")

# 3. Age impact
young_cost = df[df['age'] <= 30]['charges'].mean()
senior_cost = df[df['age'] >= 60]['charges'].mean()
age_impact = (senior_cost - young_cost) / young_cost * 100
print(f"\n3️⃣ Senior customers (60+) cost {age_impact:.1f}% more than young adults (≤30).")

# 4. Regional variation
region_cost = df.groupby('region')['charges'].mean().sort_values()
print(f"\n4️⃣ Regional cost ranking (lowest to highest):")
for reg, cost in region_cost.items():
    print(f"   {reg.title()}: ${cost:,.0f}")

# 5. Children impact
with_kids = df[df['children'] > 0]['charges'].mean()
no_kids = df[df['children'] == 0]['charges'].mean()
kids_impact = (with_kids - no_kids) / no_kids * 100
print(f"\n5️⃣ Customers with children have {kids_impact:.1f}% higher charges.")
print(f"   No children: ${no_kids:,.0f} | With children: ${with_kids:,.0f}")

# 6. Interaction: smoker + obese
smoker_obese = df[(df['smoker']=='yes') & (df['bmi']>=30)]['charges'].mean()
non_smoker_normal = df[(df['smoker']=='no') & (df['bmi']<25)]['charges'].mean()
interaction_impact = (smoker_obese - non_smoker_normal) / non_smoker_normal * 100
print(f"\n6️⃣ Smoker + obese customers pay {interaction_impact:.1f}% more than non-smoker + normal weight.")
print(f"   Non-smoker/normal: ${non_smoker_normal:,.0f} | Smoker/obese: ${smoker_obese:,.0f}")

# Save insights to text
with open('reports/business_insights.txt', 'w') as f:
    f.write("INSURANCE COST PREDICTION - BUSINESS INSIGHTS\n")
    f.write("="*50 + "\n")
    f.write(f"1. Smoker premium increase: {premium_increase:.1f}%\n")
    f.write(f"2. Obesity premium increase: {obese_impact:.1f}%\n")
    f.write(f"3. Senior premium increase: {age_impact:.1f}%\n")
    f.write(f"4. Regional cost ranking: {region_cost.to_dict()}\n")
    f.write(f"5. Children premium increase: {kids_impact:.1f}%\n")
    f.write(f"6. Smoker+obese premium increase: {interaction_impact:.1f}%\n")
print("\n✅ Business insights saved to reports/business_insights.txt")

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