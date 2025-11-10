import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, f1_score, classification_report
import os

# --- Configuration ---
RANDOM_STATE = 42
X_TRAIN_CSV = "X_train.csv"
X_TEST_CSV = "X_test.csv"
Y_TRAIN_CSV = "y_train.csv"
Y_TEST_CSV = "y_test.csv"
MODEL_FILE = "model_random_forest.joblib"  # The high-performing model to save
FEATURE_NAMES_FILE = 'feature_names.joblib'

# 1. Load Data
X_train = pd.read_csv(X_TRAIN_CSV)
X_test = pd.read_csv(X_TEST_CSV)
y_train = pd.read_csv(Y_TRAIN_CSV).squeeze()
y_test = pd.read_csv(Y_TEST_CSV).squeeze()

X_train_np = X_train.values
X_test_np = X_test.values
y_train_np = y_train.values
y_test_np = y_test.values
feature_names = X_train.columns.tolist()

print(f"Loaded training data shape: {X_train.shape}")

# 2. Define Models
models = {
    "LogisticRegression": LogisticRegression(
        random_state=RANDOM_STATE,
        solver='liblinear',
        class_weight='balanced'
    ),
    "RandomForestClassifier": RandomForestClassifier(
        random_state=RANDOM_STATE,
        n_estimators=100,
        max_depth=5,
        class_weight='balanced'
    )
}

# 3. Training and Evaluation Loop
results = {}
best_auc = 0
best_model_name = ""
best_model = None

print("\n--- Model Training & Evaluation ---")

for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train_np, y_train_np)

    # Predict probabilities (for ROC AUC)
    y_pred_proba = model.predict_proba(X_test_np)[:, 1]
    # Predict class (for F1, Precision, Recall)
    y_pred = model.predict(X_test_np)

    # Calculate metrics
    auc = roc_auc_score(y_test_np, y_pred_proba)
    f1 = f1_score(y_test_np, y_pred)

    print(f"  ROC AUC: {auc:.4f}")
    print(f"  F1 Score: {f1:.4f}")

    # Store results
    results[name] = {
        'AUC': auc,
        'F1': f1,
        'report': classification_report(y_test_np, y_pred, target_names=['No PCOS', 'PCOS'])
    }

    # Check for best model
    if auc > best_auc:
        best_auc = auc
        best_model_name = name
        best_model = model

# 4. Save Best Model and Feature Names
joblib.dump(best_model, MODEL_FILE)
print(f"\n✅ Saved best model ({best_model_name}, AUC: {best_auc:.4f}) to: {MODEL_FILE}")

# Save feature names, required for Anchor explainer initialization
joblib.dump(feature_names, FEATURE_NAMES_FILE)
print(f"Saved feature names to: {FEATURE_NAMES_FILE}")

# 5. Summary Report
print("\n--- Final Classification Reports (Test Set) ---")
for name, res in results.items():
    print(f"\nReport for {name}:")
    print(res['report'])