# ale_explainer_custom.py
"""
Manual implementation of ALE (Accumulated Local Effects)
Why Manual? -> Avoids dependency hell with 'alibi' or 'PyALE' and works
               seamlessly with your existing scikit-learn pipeline.
"""

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import os
import sys

# =================================================
# 🔧 CRITICAL: Custom function must be defined FIRST
# =================================================
def clip_nonnegative(X):
    X = np.array(X, dtype=float)
    X[X < 0] = 0.0
    return X

# ---------- CONFIGURATION ----------
CLEANED_CSV = "PCOS_infertility_cleaned.csv"
PIPELINE_FILE = "preprocessing_pipeline.joblib"
MODEL_FILE = "model_random_forest.joblib"

# The features used in training
FEATURES = ["i___beta_hcgmiu_ml", "ii____beta_hcgmiu_ml", "amhng_ml"]

# CHANGE THIS to analyze different features
FEATURE_TO_ANALYZE = "amhng_ml"
N_BINS = 20
OUT_FILE = f"ale_plot_{FEATURE_TO_ANALYZE}.png"

# 1. Load Assets
print("📦 Loading dataset, pipeline, and model...")
if not os.path.exists(PIPELINE_FILE):
    print(f"❌ Error: {PIPELINE_FILE} not found.")
    sys.exit(1)

try:
    df = pd.read_csv(CLEANED_CSV)
    preprocessor = joblib.load(PIPELINE_FILE)
    model = joblib.load(MODEL_FILE)
except Exception as e:
    print(f"❌ Load Error: {e}")
    sys.exit(1)

X = df[FEATURES].copy()
print(f"Loaded {len(X)} samples.")

# 2. Safety Checks & Binning
print(f"📊 Computing ALE values for feature: {FEATURE_TO_ANALYZE}...")

# Handle NaNs
if X[FEATURE_TO_ANALYZE].isnull().any():
    print("⚠️ Warning: NaNs detected. Filling with median.")
    X[FEATURE_TO_ANALYZE] = X[FEATURE_TO_ANALYZE].fillna(X[FEATURE_TO_ANALYZE].median())

values = X[FEATURE_TO_ANALYZE].values

# Handle Constant Features
unique_vals = np.unique(values)
if len(unique_vals) < 2:
    print(f"❌ Error: Feature '{FEATURE_TO_ANALYZE}' is constant. ALE cannot be computed.")
    sys.exit(1)

# Create Quantile Bins (ensures each bin has data)
n_bins_actual = min(N_BINS, len(unique_vals) - 1)
bins = np.unique(np.quantile(values, np.linspace(0, 1, n_bins_actual + 1)))

print(f"Created {len(bins)-1} intervals.")

ale_effects = []
bin_centers = []

# 3. Main Loop: Compute Local Effects
for i in range(len(bins) - 1):
    lower, upper = bins[i], bins[i + 1]

    # Select data points falling into this bin
    if i == len(bins) - 2:
        mask = (values >= lower) & (values <= upper)
    else:
        mask = (values >= lower) & (values < upper)

    # Skip empty bins
    if mask.sum() == 0:
        continue

    # Create synthetic copies
    X_low = X.loc[mask].copy()
    X_high = X_low.copy()

    # Force the feature value to the bin edges
    X_low[FEATURE_TO_ANALYZE] = lower
    X_high[FEATURE_TO_ANALYZE] = upper

    # Transform & Predict
    try:
        # We need the probability of class 1 (PCOS)
        pred_low = model.predict_proba(preprocessor.transform(X_low))[:, 1]
        pred_high = model.predict_proba(preprocessor.transform(X_high))[:, 1]
    except ValueError as e:
        print(f"⚠️ Prediction error in bin {lower}-{upper}: {e}")
        continue

    # The effect is the difference in prediction
    local_effect = np.mean(pred_high - pred_low)
    ale_effects.append(local_effect)
    bin_centers.append((lower + upper) / 2)

# 4. Final Processing
if len(ale_effects) == 0:
    print("❌ Error: No effects computed. Check data.")
    sys.exit(1)

ale_effects = np.array(ale_effects)
# Accumulate the effects
ale_cumulative = np.cumsum(ale_effects)
# Center the plot at 0 (so 0 means "Average Prediction")
ale_cumulative -= np.mean(ale_cumulative)

# 5. Plotting
print("📈 Plotting ALE Curve...")
plt.figure(figsize=(8, 5))

# Main Line
plt.plot(bin_centers, ale_cumulative, marker="o", color="#1f77b4", linewidth=2, label="ALE Effect")

# Add Rug Plot (shows where data actually exists)
if len(ale_cumulative) > 0:
    # Position rug just below the lowest point
    rug_y = np.min(ale_cumulative) - (np.ptp(ale_cumulative) * 0.05 if np.ptp(ale_cumulative) > 0 else 0.01)
    plt.plot(values, np.full_like(values, rug_y), '|', color='gray', alpha=0.3, label="Patient Density")

plt.axhline(0, color="black", linestyle="--", linewidth=0.8)
plt.xlabel(f"{FEATURE_TO_ANALYZE} (Value)")
plt.ylabel("Change in PCOS Probability")
plt.title(f"ALE Plot: How {FEATURE_TO_ANALYZE} affects Prediction")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(OUT_FILE, dpi=300)
plt.close()
print(f"✅ Success! Plot saved as: {OUT_FILE}")