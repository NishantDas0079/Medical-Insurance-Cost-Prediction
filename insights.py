# -*- coding: utf-8 -*-
"""
BUSINESS INSIGHTS FOR INSURANCE COST PREDICTION
------------------------------------------------
- Loads raw insurance dataset (from ZIP / local / download)
- Computes actionable insights
- Saves results to reports/business_insights.txt

This script runs independently – no model required.
"""

import os
import sys
import zipfile
import pandas as pd
import numpy as np
from datetime import datetime

# ==================== CONFIGURATION ====================
ZIP_PATH = r"C:\Users\Nishant\Downloads\archive.zip"
DATA_DIR = "data"
REPORTS_DIR = "reports"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

# ==================== DATA LOADING (ROBUST) ====================
def load_insurance_data():
    """Load the insurance dataset from multiple possible sources."""
    
    # Priority 1: Already extracted CSV in data folder
    local_paths = [
        os.path.join(DATA_DIR, "insurance.csv"),
        os.path.join(DATA_DIR, "insurance_raw.csv"),
        os.path.join(DATA_DIR, "insurance_featured.csv")  # fallback to featured
    ]
    for path in local_paths:
        if os.path.exists(path):
            print(f"✅ Found dataset at: {path}")
            return pd.read_csv(path)
    
    # Priority 2: Extract from your ZIP file
    if os.path.exists(ZIP_PATH):
        print(f"📦 Extracting from: {ZIP_PATH}")
        with zipfile.ZipFile(ZIP_PATH, 'r') as zf:
            csv_files = [f for f in zf.namelist() if f.endswith('.csv')]
            if csv_files:
                # Prefer file with 'insurance' in name
                insurance_file = next((f for f in csv_files if 'insurance' in f.lower()), csv_files[0])
                zf.extract(insurance_file, DATA_DIR)
                extracted_path = os.path.join(DATA_DIR, os.path.basename(insurance_file))
                print(f"📄 Extracted: {extracted_path}")
                return pd.read_csv(extracted_path)
    
    # Priority 3: Download from GitHub (fallback)
    print("🌐 Downloading dataset from GitHub...")
    url = "https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv"
    try:
        df = pd.read_csv(url)
        df.to_csv(os.path.join(DATA_DIR, "insurance.csv"), index=False)
        print("✅ Download complete.")
        return df
    except Exception as e:
        print(f"❌ Failed to download: {e}")
        sys.exit(1)

# ==================== INSIGHT GENERATION ====================
def generate_insights(df):
    """Compute and return business insights as a dictionary."""
    insights = {}
    
    # 1. Smoker vs Non-smoker
    smoker_cost = df.groupby('smoker')['charges'].mean()
    premium_increase = (smoker_cost['yes'] - smoker_cost['no']) / smoker_cost['no'] * 100
    insights['smoker_premium_increase'] = round(premium_increase, 1)
    insights['avg_cost_non_smoker'] = f"${smoker_cost['no']:,.0f}"
    insights['avg_cost_smoker'] = f"${smoker_cost['yes']:,.0f}"
    
    # 2. Obesity impact
    obese_cost = df[df['bmi'] >= 30]['charges'].mean()
    normal_cost = df[(df['bmi'] >= 18.5) & (df['bmi'] < 25)]['charges'].mean()
    obese_impact = (obese_cost - normal_cost) / normal_cost * 100
    insights['obesity_premium_increase'] = round(obese_impact, 1)
    insights['avg_cost_normal_bmi'] = f"${normal_cost:,.0f}"
    insights['avg_cost_obese'] = f"${obese_cost:,.0f}"
    
    # 3. Age impact (senior vs young)
    young_cost = df[df['age'] <= 30]['charges'].mean()
    senior_cost = df[df['age'] >= 60]['charges'].mean()
    age_impact = (senior_cost - young_cost) / young_cost * 100
    insights['senior_premium_increase'] = round(age_impact, 1)
    insights['avg_cost_young'] = f"${young_cost:,.0f}"
    insights['avg_cost_senior'] = f"${senior_cost:,.0f}"
    
    # 4. Regional variation
    region_cost = df.groupby('region')['charges'].mean().sort_values()
    insights['region_ranking'] = region_cost.to_dict()
    
    # 5. Children impact
    with_kids = df[df['children'] > 0]['charges'].mean()
    no_kids = df[df['children'] == 0]['charges'].mean()
    kids_impact = (with_kids - no_kids) / no_kids * 100
    insights['children_premium_increase'] = round(kids_impact, 1)
    insights['avg_cost_no_children'] = f"${no_kids:,.0f}"
    insights['avg_cost_with_children'] = f"${with_kids:,.0f}"
    
    # 6. Combined risk: smoker + obese
    smoker_obese = df[(df['smoker'] == 'yes') & (df['bmi'] >= 30)]['charges'].mean()
    non_smoker_normal = df[(df['smoker'] == 'no') & (df['bmi'] < 25)]['charges'].mean()
    if pd.notna(smoker_obese) and pd.notna(non_smoker_normal):
        combo_impact = (smoker_obese - non_smoker_normal) / non_smoker_normal * 100
        insights['smoker_obese_premium_increase'] = round(combo_impact, 1)
        insights['avg_cost_non_smoker_normal'] = f"${non_smoker_normal:,.0f}"
        insights['avg_cost_smoker_obese'] = f"${smoker_obese:,.0f}"
    else:
        insights['smoker_obese_premium_increase'] = 'N/A'
    
    return insights

# ==================== PRINT & SAVE ====================
def display_insights(insights):
    """Pretty print insights and save to file."""
    output_lines = []
    output_lines.append("=" * 60)
    output_lines.append("💡 INSURANCE COST PREDICTION – BUSINESS INSIGHTS")
    output_lines.append(f"   Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    output_lines.append("=" * 60)
    output_lines.append("")
    
    # 1
    output_lines.append("1️⃣  SMOKER IMPACT")
    output_lines.append(f"   • Smokers pay {insights['smoker_premium_increase']}% higher premiums on average.")
    output_lines.append(f"   • Non‑smoker average: {insights['avg_cost_non_smoker']}")
    output_lines.append(f"   • Smoker average:     {insights['avg_cost_smoker']}")
    output_lines.append("")
    
    # 2
    output_lines.append("2️⃣  OBESITY IMPACT")
    output_lines.append(f"   • Obese customers (BMI ≥ 30) have {insights['obesity_premium_increase']}% higher charges than normal weight.")
    output_lines.append(f"   • Normal weight average: {insights['avg_cost_normal_bmi']}")
    output_lines.append(f"   • Obese average:         {insights['avg_cost_obese']}")
    output_lines.append("")
    
    # 3
    output_lines.append("3️⃣  AGE IMPACT")
    output_lines.append(f"   • Senior customers (60+) pay {insights['senior_premium_increase']}% more than young adults (≤30).")
    output_lines.append(f"   • Young adult average: {insights['avg_cost_young']}")
    output_lines.append(f"   • Senior average:      {insights['avg_cost_senior']}")
    output_lines.append("")
    
    # 4
    output_lines.append("4️⃣  REGIONAL VARIATION")
    output_lines.append("   • Average charges by region (lowest to highest):")
    for region, cost in insights['region_ranking'].items():
        output_lines.append(f"     - {region.title()}: ${cost:,.0f}")
    output_lines.append("")
    
    # 5
    output_lines.append("5️⃣  CHILDREN IMPACT")
    output_lines.append(f"   • Customers with children have {insights['children_premium_increase']}% higher charges.")
    output_lines.append(f"   • No children average: {insights['avg_cost_no_children']}")
    output_lines.append(f"   • With children average: {insights['avg_cost_with_children']}")
    output_lines.append("")
    
    # 6
    output_lines.append("6️⃣  COMBINED RISK (SMOKER + OBESE)")
    if insights['smoker_obese_premium_increase'] != 'N/A':
        output_lines.append(f"   • Smoker + obese customers pay {insights['smoker_obese_premium_increase']}% more than non‑smoker + normal weight.")
        output_lines.append(f"   • Non‑smoker / normal weight: {insights['avg_cost_non_smoker_normal']}")
        output_lines.append(f"   • Smoker / obese:           {insights['avg_cost_smoker_obese']}")
    else:
        output_lines.append("   • Insufficient data for combined risk analysis.")
    output_lines.append("")
    
    output_lines.append("=" * 60)
    output_lines.append("✅ Insights saved to: reports/business_insights.txt")
    output_lines.append("=" * 60)
    
    # Print to console
    print("\n".join(output_lines))
    
    # Save to file
    with open(os.path.join(REPORTS_DIR, "business_insights.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(output_lines))
    
    return output_lines

# ==================== MAIN EXECUTION ====================
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("📊 BUSINESS INSIGHTS MODULE")
    print("=" * 60)
    
    # Load data
    df = load_insurance_data()
    print(f"📋 Dataset shape: {df.shape[0]} rows × {df.shape[1]} columns\n")
    
    # Verify required columns
    required = ['charges', 'smoker', 'bmi', 'age', 'region', 'children']
    missing = [col for col in required if col not in df.columns]
    if missing:
        print(f"❌ Missing columns: {missing}. Cannot generate insights.")
        sys.exit(1)
    
    # Generate and display insights
    insights = generate_insights(df)
    display_insights(insights)