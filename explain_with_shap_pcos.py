"""
explain_with_shap_pcos_final.py
------------------------------------
Generates SHAP explanations for the PCOS Random Forest model.

Outputs:
 - shap_outputs/shap_summary.png       (global summary plot)
 - shap_outputs/shap_patient_<i>.png   (local waterfall or bar plot)
"""

import os
import joblib
import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random

# ---------------- CONFIG ----------------
CLEANED_CSV = "PCOS_infertility_cleaned.csv"
PIPELINE_FILE = "preprocessing_pipeline.joblib"
MODEL_FILE = "model_random_forest.joblib"
OUT_DIR = "shap_outputs"

RANDOM_STATE = 42
SAMPLE_GLOBAL_N = 300
PATIENT_INDEX = 0
# ----------------------------------------

np.random.seed(RANDOM_STATE)
random.seed(RANDOM_STATE)

# helper (for legacy pipeline compatibility)
def clip_nonnegative(X):
    X = np.array(X, dtype=float)
    X[X < 0] = 0.0
    return X

# ---------------- LOAD ----------------
print("📦 Loading model + preprocessing pipeline...")
preprocessor = joblib.load(PIPELINE_FILE)
model = joblib.load(MODEL_FILE)

print("📄 Loading cleaned dataset...")
df = pd.read_csv(CLEANED_CSV)
features = ["i___beta_hcgmiu_ml", "ii____beta_hcgmiu_ml", "amhng_ml"]
target = "pcos_y_n"

if not all(f in df.columns for f in features):
    raise ValueError(f"Missing expected features: {features}")

X_raw = df[features]
y = df[target] if target in df.columns else None

# ---------------- TRANSFORM ----------------
print("🔄 Transforming dataset with saved preprocessor...")
X_proc = preprocessor.transform(X_raw)
n_samples, n_features = X_proc.shape
print(f"Transformed dataset shape: {X_proc.shape}")

# auto feature names (if preprocessor changed dimensionality)
if n_features != len(features):
    feature_names = [f"feature_{i}" for i in range(n_features)]
    print(f"⚠️ Adjusted feature names: {feature_names}")
else:
    feature_names = features

os.makedirs(OUT_DIR, exist_ok=True)

# ---------------- BUILD EXPLAINER ----------------
print("🧠 Building SHAP Explainer...")

masker = shap.maskers.Independent(X_proc, max_samples=300)
try:
    explainer = shap.Explainer(
        model,
        masker=masker,
        feature_names=feature_names,
        model_output="probability"
    )
    print("✅ Using SHAP.Explainer(model_output='probability')")
except Exception as e1:
    print("⚠️ Explainer() failed, trying TreeExplainer fallback:", e1)
    explainer = shap.TreeExplainer(model, feature_perturbation="interventional")

# ---------------- GLOBAL EXPLANATION ----------------
n_global = min(SAMPLE_GLOBAL_N, X_proc.shape[0])
X_sample = X_proc[np.random.choice(X_proc.shape[0], n_global, replace=False)]

print(f"🌍 Generating SHAP global explanation on {n_global} samples...")
try:
    shap_values = explainer(X_sample, check_additivity=False)
except Exception as e:
    print("⚠️ SHAP call failed:", e)
    shap_values = explainer(X_sample)

vals = getattr(shap_values, "values", shap_values)
vals = np.array(vals)

if vals.ndim == 3:  # e.g. (n, 2, features)
    vals = vals[:, 1, :]  # positive class (PCOS)
    print("ℹ️ Multi-class SHAP detected → using positive class (PCOS).")

if vals.shape[1] != X_sample.shape[1]:
    print(f"⚠️ Fixing mismatch: SHAP {vals.shape} → data {X_sample.shape}")
    vals = vals[:, :X_sample.shape[1]]

fig = plt.figure(figsize=(8, 5))
shap.summary_plot(vals, X_sample, feature_names=feature_names, show=False)
plt.tight_layout()
summary_path = os.path.join(OUT_DIR, "shap_summary.png")
plt.savefig(summary_path)
plt.close()
print(f"✅ Saved global summary to: {summary_path}")

# ---------------- LOCAL EXPLANATION ----------------
print(f"👩‍⚕️ Generating local explanation for patient index {PATIENT_INDEX}...")
if PATIENT_INDEX >= X_proc.shape[0]:
    raise IndexError(f"Patient index {PATIENT_INDEX} out of range (max {X_proc.shape[0]-1}).")

sample = X_proc[PATIENT_INDEX].reshape(1, -1)

try:
    sample_expl = explainer(sample, check_additivity=False)
except Exception as e:
    print("⚠️ Fallback without additivity flag:", e)
    sample_expl = explainer(sample)

# Extract correct class
if hasattr(sample_expl, "values"):
    vals_local = np.array(sample_expl.values)
    if vals_local.ndim == 3:
        vals_local = vals_local[0, 1, :]  # positive class
    elif vals_local.ndim == 2:
        vals_local = vals_local[0]
    else:
        vals_local = vals_local.ravel()
else:
    vals_local = np.array(sample_expl).ravel()

base_val = getattr(sample_expl, "base_values", [0])[0]
explanation = shap.Explanation(
    values=vals_local,
    base_values=base_val,
    data=sample[0],
    feature_names=feature_names
)

try:
    fig = plt.figure(figsize=(8, 4))
    shap.plots.waterfall(explanation, show=False)
    plt.tight_layout()
    local_path = os.path.join(OUT_DIR, f"shap_patient_{PATIENT_INDEX}.png")
    plt.savefig(local_path)
    plt.close()
    print(f"✅ Saved SHAP local waterfall to: {local_path}")
except Exception as e:
    print("⚠️ Waterfall failed, using bar chart:", e)
    order = np.argsort(-np.abs(vals_local))
    plt.figure(figsize=(6, 3))
    plt.bar([feature_names[i] for i in order], vals_local[order])
    plt.axhline(0, color="k", linewidth=0.6)
    plt.title(f"SHAP local contributions (Patient {PATIENT_INDEX})")
    plt.tight_layout()
    local_path = os.path.join(OUT_DIR, f"shap_patient_{PATIENT_INDEX}.png")
    plt.savefig(local_path)
    plt.close()
    print(f"✅ Saved fallback bar chart to: {local_path}")

print("\n🎯 Done. All SHAP outputs saved in:", OUT_DIR)
