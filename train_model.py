"""
Training Model untuk Heart Disease Prediction
Menggunakan fitur terpilih dan menyimpan model ke pickle
"""

import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, confusion_matrix, classification_report)
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# 1. LOAD DATA
# ==========================================
print("="*60)
print("HEART DISEASE PREDICTION - MODEL TRAINING")
print("="*60)

print("\n[1/6] Loading data...")
column_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
                'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']

data_url = "http://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
df = pd.read_csv(data_url, names=column_names, na_values='?')

# Handle missing values
df['ca'].fillna(df['ca'].mode()[0], inplace=True)
df['thal'].fillna(df['thal'].mode()[0], inplace=True)

# Convert target to binary
df['target'] = (df['target'] > 0).astype(int)

print(f"✓ Dataset loaded: {df.shape}")
print(f"✓ Target distribution: {df['target'].value_counts().to_dict()}")

# ==========================================
# 2. LOAD SELECTED FEATURES
# ==========================================
print("\n[2/6] Loading selected features...")

# Load dari pickle jika ada, atau gunakan default
try:
    with open('selected_features.pkl', 'rb') as f:
        SELECTED_FEATURES = pickle.load(f)
    print(f"✓ Loaded {len(SELECTED_FEATURES)} features from pickle")
except:
    # Default features jika file tidak ada
    SELECTED_FEATURES = ['cp', 'thalach', 'exang', 'oldpeak', 'ca', 'thal']
    print(f"✓ Using default {len(SELECTED_FEATURES)} features")

print(f"Features: {SELECTED_FEATURES}")

# ==========================================
# 3. PREPARE DATA
# ==========================================
print("\n[3/6] Preparing data...")

X = df[SELECTED_FEATURES]
y = df['target']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"✓ Train set: {X_train.shape}")
print(f"✓ Test set: {X_test.shape}")

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("✓ Features scaled")

# ==========================================
# 4. TRAIN MODELS
# ==========================================
print("\n[4/6] Training models...")

models = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
}

results = {}

for name, model in models.items():
    print(f"\n  Training {name}...")
    
    # Train
    if name == 'Logistic Regression':
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
    
    # Evaluate
    results[name] = {
        'model': model,
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'confusion_matrix': confusion_matrix(y_test, y_pred),
        'y_pred': y_pred
    }
    
    print(f"  ✓ {name} trained")

# ==========================================
# 5. EVALUATE & COMPARE
# ==========================================
print("\n[5/6] Evaluating models...")
print("\n" + "="*60)
print("MODEL COMPARISON")
print("="*60)

comparison_df = pd.DataFrame({
    'Model': list(results.keys()),
    'Accuracy': [results[m]['accuracy'] for m in results],
    'Precision': [results[m]['precision'] for m in results],
    'Recall': [results[m]['recall'] for m in results],
    'F1-Score': [results[m]['f1'] for m in results]
})

print("\n", comparison_df.to_string(index=False))

# Detailed results for each model
for name, res in results.items():
    print(f"\n{'='*60}")
    print(f"{name.upper()} - DETAILED RESULTS")
    print(f"{'='*60}")
    print(f"Accuracy:  {res['accuracy']:.4f}")
    print(f"Precision: {res['precision']:.4f}")
    print(f"Recall:    {res['recall']:.4f}")
    print(f"F1-Score:  {res['f1']:.4f}")
    
    print(f"\nConfusion Matrix:")
    print(res['confusion_matrix'])
    
    print(f"\nClassification Report:")
    print(classification_report(y_test, res['y_pred'], 
                                target_names=['No Disease', 'Disease']))

# Visualize confusion matrices
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for idx, (name, res) in enumerate(results.items()):
    sns.heatmap(res['confusion_matrix'], annot=True, fmt='d', 
                cmap='Blues', ax=axes[idx])
    axes[idx].set_title(f'{name}\nAccuracy: {res["accuracy"]:.3f}')
    axes[idx].set_ylabel('True Label')
    axes[idx].set_xlabel('Predicted Label')

plt.tight_layout()
plt.savefig('confusion_matrices.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved confusion matrices to 'confusion_matrices.png'")

# ==========================================
# 6. SAVE BEST MODEL
# ==========================================
print("\n[6/6] Saving best model...")

# Select best model based on F1-score (balance of precision and recall)
best_model_name = max(results, key=lambda x: results[x]['f1'])
best_model = results[best_model_name]['model']

print(f"\n✓ Best model: {best_model_name}")
print(f"  F1-Score: {results[best_model_name]['f1']:.4f}")

# Create models directory
import os
os.makedirs('models', exist_ok=True)

# Save model
with open('models/best_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)
print("✓ Saved model to 'models/best_model.pkl'")

# Save scaler (only if Logistic Regression is best)
if best_model_name == 'Logistic Regression':
    with open('models/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    print("✓ Saved scaler to 'models/scaler.pkl'")

# Save feature names
with open('models/feature_names.pkl', 'wb') as f:
    pickle.dump(SELECTED_FEATURES, f)
print("✓ Saved feature names to 'models/feature_names.pkl'")

# Save model info
model_info = {
    'model_name': best_model_name,
    'features': SELECTED_FEATURES,
    'accuracy': results[best_model_name]['accuracy'],
    'precision': results[best_model_name]['precision'],
    'recall': results[best_model_name]['recall'],
    'f1_score': results[best_model_name]['f1'],
    'use_scaler': best_model_name == 'Logistic Regression'
}

with open('models/model_info.pkl', 'wb') as f:
    pickle.dump(model_info, f)
print("✓ Saved model info to 'models/model_info.pkl'")

# ==========================================
# SUMMARY
# ==========================================
print("\n" + "="*60)
print("TRAINING COMPLETE!")
print("="*60)
print(f"\nBest Model: {best_model_name}")
print(f"Features used: {len(SELECTED_FEATURES)}")
print(f"Accuracy: {results[best_model_name]['accuracy']:.4f}")
print(f"F1-Score: {results[best_model_name]['f1']:.4f}")
print("\nModel files saved in 'models/' directory:")
print("  - best_model.pkl")
print("  - feature_names.pkl")
print("  - model_info.pkl")
if best_model_name == 'Logistic Regression':
    print("  - scaler.pkl")
print("\nReady for deployment! 🚀")
