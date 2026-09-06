"""
explain_with_shap_pcos_final.py
------------------------------------
Generates SHAP explanations for the PCOS Random Forest model.
Fixes: Shape mismatch errors for Random Forest (3D output).

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

# helper function (Must be defined for joblib to load the pipeline)
def clip_nonnegative(X):
    X = np.array(X, dtype=float)
    X[X < 0] = 0.0
    return X

# ---------------- LOAD ----------------
print("📦 Loading model + preprocessing pipeline...")
try:
    preprocessor = joblib.load(PIPELINE_FILE)
    model = joblib.load(MODEL_FILE)
except FileNotFoundError as e:
    print(f"❌ Error: {e}")
    print("Make sure .joblib files are in the same directory.")
    exit()

print("📄 Loading cleaned dataset...")
df = pd.read_csv(CLEANED_CSV)
features = ["i___beta_hcgmiu_ml", "ii____beta_hcgmiu_ml", "amhng_ml"]
target = "pcos_y_n"

if not all(f in df.columns for f in features):
    raise ValueError(f"Missing expected features: {features}")

X_raw = df[features]

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

# ---------------- GLOBAL EXPLANATION (FIXED) ----------------
n_global = min(SAMPLE_GLOBAL_N, X_proc.shape[0])
X_sample = X_proc[np.random.choice(X_proc.shape[0], n_global, replace=False)]

print(f"🌍 Generating SHAP global explanation on {n_global} samples...")
try:
    shap_values = explainer(X_sample, check_additivity=False)
except Exception as e:
    print("⚠️ SHAP call failed:", e)
    shap_values = explainer(X_sample)

# Extract raw values
vals = getattr(shap_values, "values", shap_values)
vals = np.array(vals)

print(f"ℹ️ Raw SHAP shape: {vals.shape}")

# === FIX: Handle 3D Output (Samples, Features, Classes) ===
if vals.ndim == 3:
    # Check if the last dimension is classes (usually 2 for binary classification)
    if vals.shape[2] == 2:
        print("ℹ️ Multi-class SHAP detected (Rows, Features, Classes) -> Selecting Class 1 (PCOS)")
        vals = vals[:, :, 1]  # Select Positive Class
    elif vals.shape[1] == 2:
        # Rare case: (Rows, Classes, Features)
        print("ℹ️ Multi-class SHAP detected (Rows, Classes, Features) -> Selecting Class 1 (PCOS)")
        vals = vals[:, 1, :]

# Verify shapes match
if vals.shape != X_sample.shape:
    print(f"⚠️ Shape mismatch detected! SHAP: {vals.shape}, Data: {X_sample.shape}")
    # Force alignment if dimensions are swapped
    if vals.shape == (X_sample.shape[1], X_sample.shape[0]):
        vals = vals.T

fig = plt.figure(figsize=(8, 5))
shap.summary_plot(vals, X_sample, feature_names=feature_names, show=False)
plt.tight_layout()
summary_path = os.path.join(OUT_DIR, "shap_summary.png")
plt.savefig(summary_path)
plt.close()
print(f"✅ Saved global summary to: {summary_path}")

# ---------------- LOCAL EXPLANATION (FIXED) ----------------
print(f"👩‍⚕️ Generating local explanation for patient index {PATIENT_INDEX}...")

sample = X_proc[PATIENT_INDEX].reshape(1, -1)

try:
    sample_expl = explainer(sample, check_additivity=False)
except Exception:
    sample_expl = explainer(sample)

# Extract values for single sample
if hasattr(sample_expl, "values"):
    vals_local = np.array(sample_expl.values)
else:
    vals_local = np.array(sample_expl)

# === FIX: Handle 3D Output for Local Sample ===
if vals_local.ndim == 3:
    # (1, Features, Classes) -> Select Class 1
    vals_local = vals_local[0, :, 1]
elif vals_local.ndim == 2:
    if vals_local.shape[0] == 1:
         # (1, Features) -> Flatten
        vals_local = vals_local[0]
    else:
        # (Features, Classes) -> Select Class 1
        vals_local = vals_local[:, 1]

# Get base value (expected value)
base_val = getattr(sample_expl, "base_values", [0])
if isinstance(base_val, np.ndarray):
    if base_val.ndim > 1 and base_val.shape[-1] == 2:
        base_val = base_val[0, 1]  # Class 1 base value
    else:
        base_val = base_val.ravel()[0]
elif isinstance(base_val, list):
    base_val = base_val[0]

# Construct Explanation Object
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