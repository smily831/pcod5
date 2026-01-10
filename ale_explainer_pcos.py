"""
ALE (Accumulated Local Effects) for PCOS Prediction
--------------------------------------------------
Manual ALE implementation (NO pyALE dependency)

Fixes:
- Handles custom clip_nonnegative function used in pipeline
- Fully compatible with saved preprocessing_pipeline.joblib
"""

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# =================================================
# 🔧 IMPORTANT: custom function MUST be defined FIRST
# =================================================
def clip_nonnegative(X):
    X = np.array(X, dtype=float)
    X[X < 0] = 0.0
    return X

# ---------- CONFIG ----------
CLEANED_CSV = "PCOS_infertility_cleaned.csv"
PIPELINE_FILE = "preprocessing_pipeline.joblib"
MODEL_FILE = "model_random_forest.joblib"

FEATURES = ["i___beta_hcgmiu_ml", "ii____beta_hcgmiu_ml", "amhng_ml"]
FEATURE_TO_ANALYZE = "amhng_ml"

N_BINS = 10
OUT_FILE = "ale_amh_pcos.png"
# ---------------------------

print("📦 Loading dataset, pipeline, and model...")

# Load artifacts AFTER defining clip_nonnegative
df = pd.read_csv(CLEANED_CSV)
preprocessor = joblib.load(PIPELINE_FILE)
model = joblib.load(MODEL_FILE)

X = df[FEATURES].copy()

# ---------- ALE COMPUTATION ----------
print("📊 Computing ALE values...")

values = X[FEATURE_TO_ANALYZE].values

# Step 1: Quantile-based bins (ALE requirement)
bins = np.quantile(values, np.linspace(0, 1, N_BINS + 1))
bins = np.unique(bins)

ale_effects = []
bin_centers = []

# Step 2: Compute local effects
for i in range(len(bins) - 1):
    lower, upper = bins[i], bins[i + 1]

    mask = (values >= lower) & (values < upper)
    if mask.sum() == 0:
        continue

    X_low = X.loc[mask].copy()
    X_high = X_low.copy()

    X_low[FEATURE_TO_ANALYZE] = lower
    X_high[FEATURE_TO_ANALYZE] = upper

    pred_low = model.predict_proba(
        preprocessor.transform(X_low)
    )[:, 1]

    pred_high = model.predict_proba(
        preprocessor.transform(X_high)
    )[:, 1]

    local_effect = np.mean(pred_high - pred_low)

    ale_effects.append(local_effect)
    bin_centers.append((lower + upper) / 2)

# Step 3: Accumulate & center
ale_effects = np.array(ale_effects)
ale_cumulative = np.cumsum(ale_effects)
ale_cumulative -= np.mean(ale_cumulative)

# ---------- PLOT ----------
print("📈 Plotting ALE...")

plt.figure(figsize=(7, 4))
plt.plot(bin_centers, ale_cumulative, marker="o", linewidth=2)
plt.axhline(0, color="black", linestyle="--", linewidth=0.8)
plt.xlabel(FEATURE_TO_ANALYZE)
plt.ylabel("ALE effect on PCOS probability")
plt.title(f"ALE Plot for {FEATURE_TO_ANALYZE}")
plt.grid(True)
plt.tight_layout()
plt.savefig(OUT_FILE)
plt.close()

print(f"✅ ALE plot saved as: {OUT_FILE}")
print("🎯 ALE explanation completed successfully.")
