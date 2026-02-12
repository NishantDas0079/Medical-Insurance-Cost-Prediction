# 💡 Business Insights
Generated from raw data – no model required.

```text
1️⃣  SMOKER IMPACT
   • Smokers pay 238.5% higher premiums on average.
   • Non‑smoker average: $8,434 | Smoker average: $32,050

2️⃣  OBESITY IMPACT
   • Obese customers (BMI ≥ 30) have 56.9% higher charges than normal weight.
   • Normal weight average: $10,710 | Obese average: $16,804

3️⃣  AGE IMPACT
   • Senior customers (60+) pay 303.1% more than young adults (≤30).
   • Young adult average: $9,424 | Senior average: $38,000

4️⃣  REGIONAL VARIATION
   • Average charges (lowest to highest):
     - Southeast: $14,735
     - Southwest: $12,346
     - Northwest: $12,417
     - Northeast: $13,414

5️⃣  CHILDREN IMPACT
   • Customers with children have 24.5% higher charges.
   • No children: $12,370 | With children: $15,400

6️⃣  COMBINED RISK (SMOKER + OBESE)
   • Smoker + obese customers pay 467.8% more than non‑smoker + normal weight.
   • Non‑smoker/normal: $9,222 | Smoker/obese: $52,367
```

These insights can directly inform:

Premium pricing strategies

Wellness program targeting

Customer segmentation

Risk assessment models

# Sample Prediction
```python
from predict import predict_charge

sample = {
    'age': 35,
    'sex': 'male',
    'bmi': 28.5,
    'children': 2,
    'smoker': 'no',
    'region': 'southeast'
}
predicted = predict_charge(**sample)
print(f"Predicted annual charges: ${predicted:,.2f}")
```
Output: `Predicted annual charges: $12,845.30`

# 📚 Dependencies
See `requirements.txt` for full list.
```text
numpy
pandas
scikit-learn
matplotlib
seaborn
xgboost
shap          # optional, for feature importance plots
joblib
```
