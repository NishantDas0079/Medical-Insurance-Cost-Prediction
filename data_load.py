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