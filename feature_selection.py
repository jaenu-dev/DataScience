"""
Feature Selection untuk Heart Disease Prediction
Menganalisis korelasi fitur dengan target dan memilih fitur terbaik
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# Load data
print("Loading data...")
column_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
                'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']

data_url = "http://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
df = pd.read_csv(data_url, names=column_names, na_values='?')

# Handle missing values
print("Handling missing values...")
df['ca'].fillna(df['ca'].mode()[0], inplace=True)
df['thal'].fillna(df['thal'].mode()[0], inplace=True)

# Convert target to binary (0 = no disease, 1 = disease)
df['target'] = (df['target'] > 0).astype(int)

print(f"\nDataset shape: {df.shape}")
print(f"Target distribution:\n{df['target'].value_counts()}")

# ==========================================
# 1. CORRELATION ANALYSIS
# ==========================================
print("\n" + "="*50)
print("CORRELATION ANALYSIS")
print("="*50)

# Calculate correlation with target
correlations = df.corr()['target'].sort_values(ascending=False)
print("\nCorrelation with target:")
print(correlations)

# Visualize top correlations
plt.figure(figsize=(10, 8))
correlations.drop('target').plot(kind='barh')
plt.title('Feature Correlation with Target (Heart Disease)')
plt.xlabel('Correlation Coefficient')
plt.tight_layout()
plt.savefig('correlation_analysis.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved correlation plot to 'correlation_analysis.png'")

# ==========================================
# 2. FEATURE IMPORTANCE (Random Forest)
# ==========================================
print("\n" + "="*50)
print("FEATURE IMPORTANCE ANALYSIS")
print("="*50)

X = df.drop('target', axis=1)
y = df['target']

# Train Random Forest for feature importance
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X, y)

# Get feature importance
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

print("\nFeature Importance (Random Forest):")
print(feature_importance)

# Visualize feature importance
plt.figure(figsize=(10, 8))
plt.barh(feature_importance['feature'], feature_importance['importance'])
plt.xlabel('Importance')
plt.title('Feature Importance (Random Forest)')
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved feature importance plot to 'feature_importance.png'")

# ==========================================
# 3. SELECT TOP FEATURES
# ==========================================
print("\n" + "="*50)
print("FEATURE SELECTION")
print("="*50)

# Metode 1: Berdasarkan correlation (absolute value)
top_corr_features = correlations.abs().sort_values(ascending=False)[1:8].index.tolist()
print("\nTop 7 features by correlation:")
for i, feat in enumerate(top_corr_features, 1):
    print(f"{i}. {feat}: {correlations[feat]:.3f}")

# Metode 2: Berdasarkan feature importance
top_importance_features = feature_importance.head(7)['feature'].tolist()
print("\nTop 7 features by importance:")
for i, row in feature_importance.head(7).iterrows():
    print(f"{i+1}. {row['feature']}: {row['importance']:.3f}")

# Kombinasi: ambil fitur yang muncul di kedua metode
selected_features = list(set(top_corr_features) & set(top_importance_features))
print(f"\nFeatures appearing in both methods: {selected_features}")

# Jika terlalu sedikit, ambil top 6 dari importance
if len(selected_features) < 5:
    selected_features = top_importance_features[:6]
    print(f"\nUsing top 6 features from importance: {selected_features}")

# ==========================================
# 4. FINAL SELECTION
# ==========================================
print("\n" + "="*50)
print("FINAL SELECTED FEATURES")
print("="*50)

# Berdasarkan analisis, pilih fitur terbaik
# Kombinasi dari correlation dan importance
FINAL_FEATURES = ['cp', 'thalach', 'exang', 'oldpeak', 'ca', 'thal']

print(f"\nFinal selected features ({len(FINAL_FEATURES)}):")
for i, feat in enumerate(FINAL_FEATURES, 1):
    corr_val = correlations[feat]
    imp_val = feature_importance[feature_importance['feature'] == feat]['importance'].values[0]
    print(f"{i}. {feat}")
    print(f"   - Correlation: {corr_val:.3f}")
    print(f"   - Importance: {imp_val:.3f}")

# Save selected features
import pickle
with open('selected_features.pkl', 'wb') as f:
    pickle.dump(FINAL_FEATURES, f)
print("\n✓ Saved selected features to 'selected_features.pkl'")

# ==========================================
# 5. FEATURE DESCRIPTIONS
# ==========================================
print("\n" + "="*50)
print("FEATURE DESCRIPTIONS")
print("="*50)

feature_descriptions = {
    'cp': 'Chest Pain Type (1-4)',
    'thalach': 'Maximum Heart Rate Achieved',
    'exang': 'Exercise Induced Angina (0/1)',
    'oldpeak': 'ST Depression Induced by Exercise',
    'ca': 'Number of Major Vessels (0-3)',
    'thal': 'Thalassemia (3=normal, 6=fixed defect, 7=reversible defect)'
}

print("\nSelected features explanation:")
for feat in FINAL_FEATURES:
    print(f"- {feat}: {feature_descriptions.get(feat, 'N/A')}")

print("\n" + "="*50)
print("FEATURE SELECTION COMPLETE!")
print("="*50)
print(f"\nReduced from {len(X.columns)} features to {len(FINAL_FEATURES)} features")
print("Ready for model training with selected features.")
