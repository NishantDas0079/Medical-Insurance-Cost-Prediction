# -*- coding: utf-8 -*-
"""
Created on Thu Feb 12 00:04:06 2026

@author: Nishant
"""

# -*- coding: utf-8 -*-
"""
MEDICAL INSURANCE COST PREDICTION
==================================
Dataset: Medical Cost Personal (Kaggle)
Target : charges (continuous) → Regression problem
Author : Your Name
Date   : 2026-02-11
"""

# ==================== 1. IMPORTS ====================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
import os
import zipfile
import joblib
from datetime import datetime

# Preprocessing & Modeling
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb

# Optional: SHAP for interpretability (install: pip install shap)
try:
    import shap
    SHAP_AVAILABLE = True
except:
    SHAP_AVAILABLE = False

# ==================== 2. SETUP ====================
# Create directories
os.makedirs('data', exist_ok=True)
os.makedirs('models', exist_ok=True)
os.makedirs('reports', exist_ok=True)
os.makedirs('reports/figures', exist_ok=True)

# Plot style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('husl')
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

print("✅ Setup complete!")
print(f"📅 Started at: {datetime.now().strftime('%H:%M:%S')}\n")

# ==================== 3. DATA LOADING ====================
def load_insurance_data():
    """Loads insurance.csv from your archive.zip"""
    zip_path = r"C:\Users\Nishant\Downloads\archive.zip"
    
    if os.path.exists(zip_path):
        print(f"📦 Found archive: {zip_path}")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # List files in zip
            file_list = zip_ref.namelist()
            print(f"📋 Files in archive: {file_list}")
            
            # Find the insurance CSV
            csv_file = None
            for f in file_list:
                if 'insurance' in f.lower() and f.endswith('.csv'):
                    csv_file = f
                    break
            if csv_file is None:
                # take any .csv
                csv_file = [f for f in file_list if f.endswith('.csv')][0]
            
            print(f"📄 Extracting: {csv_file}")
            zip_ref.extract(csv_file, 'data/')
            
            # Full path to extracted CSV
            csv_path = os.path.join('data', os.path.basename(csv_file))
            df = pd.read_csv(csv_path)
            print(f"✅ Loaded: {csv_path}")
            print(f"   Shape: {df.shape[0]} rows × {df.shape[1]} columns")
            return df
    else:
        print(f"❌ Archive not found at: {zip_path}")
        print("📥 Attempting to download from Kaggle URL...")
        # Fallback: download directly (if internet available)
        url = "https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv"
        df = pd.read_csv(url)
        df.to_csv('data/insurance.csv', index=False)
        print("✅ Downloaded and saved to data/insurance.csv")
        return df

df = load_insurance_data()

# ==================== 4. EXPLORATORY DATA ANALYSIS ====================
def explore_data(df):
    print("\n" + "="*60)
    print("🔍 EXPLORATORY DATA ANALYSIS")
    print("="*60)
    
    print("\n📋 First 5 rows:")
    print(df.head())
    
    print("\n📊 Data Info:")
    print(df.info())
    
    print("\n📈 Statistical Summary:")
    print(df.describe().T)
    
    # Check missing values
    missing = df.isnull().sum()
    if missing.sum() == 0:
        print("\n✅ No missing values.")
    else:
        print("\n⚠️ Missing values:")
        print(missing[missing > 0])
    
    # Target distribution
    print("\n💰 Target variable 'charges' distribution:")
    print(f"   Mean:  ${df['charges'].mean():,.2f}")
    print(f"   Median: ${df['charges'].median():,.2f}")
    print(f"   Std:   ${df['charges'].std():,.2f}")
    print(f"   Min:   ${df['charges'].min():,.2f}")
    print(f"   Max:   ${df['charges'].max():,.2f}")
    
    # Skewness
    skew = df['charges'].skew()
    print(f"   Skewness: {skew:.2f} (right‑skewed → log transform helps)")
    
    # Categorical columns
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    print("\n📌 Categorical features:", cat_cols)
    for col in cat_cols:
        print(f"   {col}: {df[col].nunique()} unique values")
        print(df[col].value_counts().to_string(index=True))
        print()
    
    return df

df = explore_data(df)

# ==================== 5. VISUALIZATION ====================
def plot_eda(df):
    print("\n" + "="*60)
    print("📊 GENERATING VISUALIZATIONS")
    print("="*60)
    
    # 1. Distribution of charges
    fig, axes = plt.subplots(1, 2, figsize=(14,5))
    sns.histplot(df['charges'], kde=True, ax=axes[0], color='steelblue')
    axes[0].set_title('Distribution of Medical Charges', fontsize=14)
    axes[0].set_xlabel('Charges ($)')
    
    sns.boxplot(x=df['charges'], ax=axes[1], color='coral')
    axes[1].set_title('Boxplot of Charges', fontsize=14)
    axes[1].set_xlabel('Charges ($)')
    plt.tight_layout()
    plt.savefig('reports/figures/charges_distribution.png', dpi=150)
    plt.show()
    
    # 2. Charges by smoker
    plt.figure(figsize=(8,6))
    sns.boxplot(x='smoker', y='charges', data=df, palette='Set2')
    plt.title('Medical Charges by Smoking Status', fontsize=14)
    plt.ylabel('Charges ($)')
    plt.xlabel('Smoker')
    plt.tight_layout()
    plt.savefig('reports/figures/charges_by_smoker.png', dpi=150)
    plt.show()
    
    # 3. Age vs Charges colored by smoker
    plt.figure(figsize=(10,6))
    sns.scatterplot(x='age', y='charges', hue='smoker', data=df, alpha=0.7, s=60)
    plt.title('Age vs Charges (by Smoker)', fontsize=14)
    plt.xlabel('Age')
    plt.ylabel('Charges ($)')
    plt.tight_layout()
    plt.savefig('reports/figures/age_vs_charges.png', dpi=150)
    plt.show()
    
    # 4. BMI vs Charges colored by smoker
    plt.figure(figsize=(10,6))
    sns.scatterplot(x='bmi', y='charges', hue='smoker', data=df, alpha=0.7, s=60)
    plt.title('BMI vs Charges (by Smoker)', fontsize=14)
    plt.xlabel('BMI')
    plt.ylabel('Charges ($)')
    plt.tight_layout()
    plt.savefig('reports/figures/bmi_vs_charges.png', dpi=150)
    plt.show()
    
    # 5. Correlation heatmap (numeric only)
    numeric_df = df.select_dtypes(include=[np.number])
    plt.figure(figsize=(8,6))
    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title('Correlation Matrix', fontsize=14)
    plt.tight_layout()
    plt.savefig('reports/figures/correlation_heatmap.png', dpi=150)
    plt.show()
    
    # 6. Pairplot for key variables
    sns.pairplot(df[['age', 'bmi', 'children', 'charges', 'smoker']], hue='smoker')
    plt.savefig('reports/figures/pairplot.png', dpi=150)
    plt.show()
    
    print("✅ Visualizations saved to reports/figures/")

plot_eda(df)

# ==================== 6. FEATURE ENGINEERING ====================
def feature_engineering(df):
    print("\n" + "="*60)
    print("🛠️  FEATURE ENGINEERING")
    print("="*60)
    
    df_fe = df.copy()
    
    # 1. BMI categories
    df_fe['bmi_category'] = pd.cut(df_fe['bmi'], 
                                   bins=[0, 18.5, 25, 30, 100],
                                   labels=['Underweight', 'Normal', 'Overweight', 'Obese'])
    
    # 2. Age groups
    df_fe['age_group'] = pd.cut(df_fe['age'], 
                                bins=[0, 30, 45, 60, 100],
                                labels=['<30', '30-45', '45-60', '60+'])
    
    # 3. Interaction: smoker & bmi, smoker & age
    df_fe['smoker_bmi'] = df_fe['smoker'].map({'yes':1,'no':0}) * df_fe['bmi']
    df_fe['smoker_age'] = df_fe['smoker'].map({'yes':1,'no':0}) * df_fe['age']
    
    # 4. Family size indicator
    df_fe['has_children'] = (df_fe['children'] > 0).astype(int)
    
    # 5. Log transformation of target (for models sensitive to skewness)
    df_fe['log_charges'] = np.log1p(df_fe['charges'])   # log(1+charges)
    
    print("✅ Created new features:")
    print(f"   - bmi_category (4 groups)")
    print(f"   - age_group (4 groups)")
    print(f"   - smoker_bmi (interaction)")
    print(f"   - smoker_age (interaction)")
    print(f"   - has_children (binary)")
    print(f"   - log_charges (transformed target)")
    
    return df_fe

df = feature_engineering(df)

# ==================== 7. PREPARE DATA FOR MODELING ====================
# Choose target: we will predict 'log_charges' to handle skew, then back-transform
target = 'log_charges'
# If you prefer raw charges, set target = 'charges' and comment log transform steps

X = df.drop(['charges', 'log_charges'], axis=1)
y = df[target]

# Identify feature types
numerical_features = X.select_dtypes(include=[np.number]).columns.tolist()
categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

print("\n🔢 Feature sets:")
print(f"   Numerical ({len(numerical_features)}): {numerical_features}")
print(f"   Categorical ({len(categorical_features)}): {categorical_features}")

# ==================== 8. TRAIN-TEST SPLIT ====================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"\n📊 Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")

# ==================== 9. PREPROCESSING PIPELINES ====================
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

# ==================== 10. MODEL DEFINITIONS ====================
models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(alpha=1.0),
    'Lasso Regression': Lasso(alpha=0.1),
    'Decision Tree': DecisionTreeRegressor(max_depth=5, random_state=42),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
    'XGBoost': xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42, n_jobs=-1)
}

# ==================== 11. TRAINING & EVALUATION FUNCTION ====================
def evaluate_model(model, X_train, y_train, X_test, y_test, model_name):
    """Train, predict, and return metrics"""
    # Create pipeline with preprocessor + model
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', model)
    ])
    
    # Train
    pipeline.fit(X_train, y_train)
    
    # Predict
    y_pred_train = pipeline.predict(X_train)
    y_pred_test = pipeline.predict(X_test)
    
    # If we predicted log_charges, back-transform to charges
    if target == 'log_charges':
        y_train_exp = np.expm1(y_train)
        y_test_exp = np.expm1(y_test)
        y_pred_train_exp = np.expm1(y_pred_train)
        y_pred_test_exp = np.expm1(y_pred_test)
    else:
        y_train_exp = y_train
        y_test_exp = y_test
        y_pred_train_exp = y_pred_train
        y_pred_test_exp = y_pred_test
    
    # Metrics on original scale (charges)
    metrics = {
        'R2_train': r2_score(y_train_exp, y_pred_train_exp),
        'R2_test': r2_score(y_test_exp, y_pred_test_exp),
        'MAE_train': mean_absolute_error(y_train_exp, y_pred_train_exp),
        'MAE_test': mean_absolute_error(y_test_exp, y_pred_test_exp),
        'RMSE_train': np.sqrt(mean_squared_error(y_train_exp, y_pred_train_exp)),
        'RMSE_test': np.sqrt(mean_squared_error(y_test_exp, y_pred_test_exp)),
        'MAPE_test': np.mean(np.abs((y_test_exp - y_pred_test_exp) / y_test_exp)) * 100
    }
    
    return pipeline, metrics

# Train all models and collect results
results = {}
trained_pipelines = {}

print("\n" + "="*60)
print("🤖 TRAINING MODELS")
print("="*60)

for name, model in models.items():
    print(f"\n📌 {name} ...")
    pipeline, metrics = evaluate_model(model, X_train, y_train, X_test, y_test, name)
    trained_pipelines[name] = pipeline
    results[name] = metrics
    
    print(f"   R² (test): {metrics['R2_test']:.4f}")
    print(f"   MAE (test): ${metrics['MAE_test']:,.2f}")
    print(f"   RMSE(test): ${metrics['RMSE_test']:,.2f}")
    print(f"   MAPE(test): {metrics['MAPE_test']:.2f}%")

# ==================== 12. MODEL COMPARISON ====================
results_df = pd.DataFrame(results).T
results_df = results_df.sort_values('R2_test', ascending=False)
print("\n" + "="*60)
print("📈 MODEL PERFORMANCE COMPARISON")
print("="*60)
print(results_df[['R2_test', 'MAE_test', 'RMSE_test', 'MAPE_test']].round(4))

# Save results
results_df.to_csv('reports/model_performance.csv')
print("\n✅ Model performance saved to reports/model_performance.csv")

# ==================== 13. BEST MODEL SELECTION ====================
best_model_name = results_df.index[0]
best_model_pipeline = trained_pipelines[best_model_name]
print(f"\n🏆 Best model: {best_model_name}")
print(f"   R² score: {results_df.loc[best_model_name, 'R2_test']:.4f}")
print(f"   MAE: ${results_df.loc[best_model_name, 'MAE_test']:,.2f}")

# Save best model
joblib.dump(best_model_pipeline, 'models/best_insurance_model.pkl')
print(f"💾 Best model saved to models/best_insurance_model.pkl")

# ==================== 14. FEATURE IMPORTANCE (TREE MODELS) ====================
def plot_feature_importance(model_pipeline, model_name, top_n=10):
    """Extract feature names and importances if model supports it"""
    # Get the regressor from pipeline
    regressor = model_pipeline.named_steps['regressor']
    if hasattr(regressor, 'feature_importances_'):
        # Get preprocessor feature names
        preprocessor = model_pipeline.named_steps['preprocessor']
        try:
            # For one-hot encoding, get feature names
            cat_features = preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(categorical_features)
            all_features = numerical_features + list(cat_features)
        except:
            all_features = numerical_features + categorical_features
        
        importances = regressor.feature_importances_
        # Ensure length matches
        if len(importances) == len(all_features):
            feat_imp = pd.DataFrame({'Feature': all_features, 'Importance': importances})
            feat_imp = feat_imp.sort_values('Importance', ascending=False).head(top_n)
            
            plt.figure(figsize=(10,6))
            plt.barh(feat_imp['Feature'][::-1], feat_imp['Importance'][::-1], color='teal')
            plt.xlabel('Importance')
            plt.title(f'Top {top_n} Feature Importances - {model_name}')
            plt.tight_layout()
            plt.savefig(f'reports/figures/feature_importance_{model_name.replace(" ", "_")}.png', dpi=150)
            plt.show()
            return feat_imp
    return None

# Plot for best tree-based model
if any(name in best_model_name for name in ['Random Forest', 'Gradient', 'XGBoost', 'Decision']):
    print("\n🔝 Feature Importance for Best Model:")
    feat_df = plot_feature_importance(best_model_pipeline, best_model_name)
    if feat_df is not None:
        print(feat_df.to_string(index=False))

# ==================== 15. SHAP ANALYSIS (Optional) ====================
if SHAP_AVAILABLE and any(name in best_model_name for name in ['Random Forest', 'Gradient', 'XGBoost', 'Decision']):
    print("\n📊 SHAP Analysis (model interpretability)...")
    # Transform test data using preprocessor
    X_test_processed = best_model_pipeline.named_steps['preprocessor'].transform(X_test)
    model = best_model_pipeline.named_steps['regressor']
    
    # Create SHAP explainer
    if 'XGBoost' in best_model_name or 'Gradient' in best_model_name:
        explainer = shap.TreeExplainer(model)
    else:
        explainer = shap.TreeExplainer(model)  # also works for RF
    
    shap_values = explainer.shap_values(X_test_processed)
    
    # Summary plot
    plt.figure()
    shap.summary_plot(shap_values, X_test_processed, feature_names=None, show=False)
    plt.tight_layout()
    plt.savefig('reports/figures/shap_summary.png', dpi=150)
    plt.show()
    print("✅ SHAP summary plot saved.")

# ==================== 16. BUSINESS INSIGHTS ====================
print("\n" + "="*60)
print("💡 BUSINESS INSIGHTS & RECOMMENDATIONS")
print("="*60)

# Use original dataframe df (before encoding) for group analysis
df_original = pd.read_csv('data/insurance.csv')  # reload original

# Insight 1: Smoker vs Non-smoker cost difference
smoker_cost = df_original.groupby('smoker')['charges'].mean()
premium_increase = (smoker_cost['yes'] - smoker_cost['no']) / smoker_cost['no'] * 100
print(f"\n1️⃣ Smokers pay {premium_increase:.1f}% higher premiums on average.")
print(f"   Average non-smoker: ${smoker_cost['no']:,.0f}")
print(f"   Average smoker:     ${smoker_cost['yes']:,.0f}")

# Insight 2: BMI impact
obese_cost = df_original[df_original['bmi'] >= 30]['charges'].mean()
normal_cost = df_original[(df_original['bmi'] >= 18.5) & (df_original['bmi'] < 25)]['charges'].mean()
print(f"\n2️⃣ Obese customers (BMI≥30) have ${obese_cost - normal_cost:,.0f} higher average charges than normal weight.")
print(f"   That's a {((obese_cost-normal_cost)/normal_cost*100):.1f}% increase.")

# Insight 3: Age impact
young_cost = df_original[df_original['age'] <= 30]['charges'].mean()
senior_cost = df_original[df_original['age'] >= 60]['charges'].mean()
print(f"\n3️⃣ Senior customers (60+) cost {((senior_cost-young_cost)/young_cost*100):.1f}% more than young adults (≤30).")

# Insight 4: Regional variation
region_cost = df_original.groupby('region')['charges'].mean().sort_values()
print(f"\n4️⃣ Regional cost ranking (lowest to highest):")
for reg, cost in region_cost.items():
    print(f"   {reg.title()}: ${cost:,.0f}")

# Insight 5: Children impact
with_children = df_original[df_original['children'] > 0]['charges'].mean()
without_children = df_original[df_original['children'] == 0]['charges'].mean()
print(f"\n5️⃣ Customers with children have {((with_children-without_children)/without_children*100):.1f}% higher charges.")
print(f"   (${with_children:,.0f} vs ${without_children:,.0f})")

# ==================== 17. SAMPLE PREDICTION FUNCTION ====================
def predict_insurance_charge(age, sex, bmi, children, smoker, region):
    """Predict insurance charge using best model"""
    # Create DataFrame with single row
    input_df = pd.DataFrame({
        'age': [age],
        'sex': [sex],
        'bmi': [bmi],
        'children': [children],
        'smoker': [smoker],
        'region': [region]
    })
    # Apply feature engineering (must match training)
    input_df['bmi_category'] = pd.cut(input_df['bmi'], 
                                      bins=[0,18.5,25,30,100],
                                      labels=['Underweight','Normal','Overweight','Obese'])
    input_df['age_group'] = pd.cut(input_df['age'],
                                   bins=[0,30,45,60,100],
                                   labels=['<30','30-45','45-60','60+'])
    input_df['smoker_bmi'] = input_df['smoker'].map({'yes':1,'no':0}) * input_df['bmi']
    input_df['smoker_age'] = input_df['smoker'].map({'yes':1,'no':0}) * input_df['age']
    input_df['has_children'] = (input_df['children'] > 0).astype(int)
    
    # Keep only features used in training (X columns)
    input_df = input_df[X.columns]
    
    # Predict
    pred_log = best_model_pipeline.predict(input_df)[0]
    pred_charge = np.expm1(pred_log) if target == 'log_charges' else pred_log
    return pred_charge

# Example prediction
print("\n" + "="*60)
print("🔮 SAMPLE PREDICTION")
print("="*60)
sample = {
    'age': 35,
    'sex': 'male',
    'bmi': 28.5,
    'children': 2,
    'smoker': 'no',
    'region': 'southeast'
}
pred = predict_insurance_charge(**sample)
print(f"Input: {sample}")
print(f"Predicted annual medical charges: ${pred:,.2f}")

# ==================== 18. EXECUTION SUMMARY ====================
print("\n" + "="*60)
print("✅ PROJECT COMPLETED SUCCESSFULLY")
print("="*60)
print(f"📁 Outputs generated:")
print("   - reports/model_performance.csv")
print("   - reports/figures/*.png")
print("   - models/best_insurance_model.pkl")
print(f"\n⏱️  Finished at: {datetime.now().strftime('%H:%M:%S')}")