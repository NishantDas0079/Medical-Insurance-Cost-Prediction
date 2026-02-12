# 🏥 Medical Insurance Cost Prediction

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Jupyter Notebook](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3+-orange)](https://scikit-learn.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-green)](https://xgboost.ai)

## 📌 **Project Overview**

This project builds a **regression model** to predict **individual medical insurance costs** (billed charges) based on patient demographics, lifestyle factors, and regional data. The goal is to help insurance companies accurately price premiums and identify key drivers of healthcare expenses.

**Key Features:**
- 📊 Comprehensive Exploratory Data Analysis (EDA) with visualizations
- 🛠️ Advanced feature engineering (BMI categories, age groups, interaction terms)
- 🤖 7 regression models (Linear, Ridge, Lasso, Decision Tree, Random Forest, Gradient Boosting, XGBoost)
- 📈 Model evaluation using R², MAE, RMSE, MAPE
- 💡 Actionable business insights from data
- 🔮 Production-ready prediction function

---

## 📁 **Dataset**

**Source:** [Medical Cost Personal Datasets (Kaggle)](https://www.kaggle.com/mirichoi0218/insurance)  
**Size:** 1,338 records × 7 features  
**Target:** `charges` – Individual medical costs billed by health insurance.

| Feature       | Type        | Description |
|---------------|-------------|-------------|
| `age`         | int         | Age of primary beneficiary |
| `sex`         | object      | Gender (male/female) |
| `bmi`         | float       | Body mass index |
| `children`    | int         | Number of children covered |
| `smoker`      | object      | Smoking status (yes/no) |
| `region`      | object      | Residential area (NE, SE, SW, NW) |
| `charges`     | float       | **Target**: Medical costs (USD) |

---


---

## ⚙️ **Installation**

### **1. Clone the repository**
```bash
git clone https://github.com/NishantDas0079/Medical-Insurance-Cost-Prediction.git
cd Medical-Insurance-Cost-Prediction
```

# Installing dependencies
```
pip install -r requirements.txt
```

# 📊 Methodology
# 1. Exploratory Data Analysis
Distribution analysis of charges (right‑skewed → log transform applied)

Correlation heatmap

Boxplots & scatter plots to visualise relationships

# 2. Feature Engineering
BMI Categories: Underweight, Normal, Overweight, Obese

Age Groups: <30, 30–45, 45–60, 60+

Interaction Terms: smoker × bmi, smoker × age

Binary Flag: has_children

Log Transformation of target variable to handle skewness

# 3. Preprocessing Pipeline
Numerical features: StandardScaler

Categorical features: OneHotEncoder (drop first)

ColumnTransformer + Pipeline for clean, reusable code

# 4. Models Implemented
```
Model	Library
Linear Regression	scikit‑learn

Ridge Regression	scikit‑learn

Lasso Regression	scikit‑learn

Decision Tree Regressor	scikit‑learn

Random Forest Regressor	scikit‑learn

Gradient Boosting	scikit‑learn

XGBoost Regressor	XGBoost
```

# 5. Evaluation Metrics
R² Score – Proportion of variance explained

MAE – Mean Absolute Error (USD)

RMSE – Root Mean Squared Error (USD)

MAPE – Mean Absolute Percentage Error (%)

# 📈 Results
```
Model Performance Comparison

Model	R² Score	MAE ($)	RMSE ($)	MAPE (%)

Random Forest	0.86	2450	4150	32.1

XGBoost	0.85	2580	4300	34.5

Gradient Boosting	0.84	2620	4450	35.8

Decision Tree	0.71	3300	5600	42.3

Ridge Regression	0.76	4000	6100	48.2

Linear Regression	0.75	4100	6200	49.0

Lasso Regression	0.74	4200	6300	50.1

Best Model: Random Forest Regressor – selected based on highest R² and lowest error metrics.
```

# Feature Importance (Top 5)
smoker – dominant predictor

age

bmi

smoker_bmi (interaction)

children

