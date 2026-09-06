# pdp_plots.py
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.inspection import PartialDependenceDisplay

# ---------------- CONFIG ----------------
CLEANED_CSV = "PCOS_infertility_cleaned.csv"
PIPELINE_FILE = "preprocessing_pipeline.joblib"
MODEL_FILE = "model_random_forest.joblib"
# ----------------------------------------

# Helper function (Must be defined exactly as it was during training)
def clip_nonnegative(X):
    import numpy as _np
    X = _np.array(X, dtype=float)
    X[X < 0] = 0.0
    return X

# 1️⃣ Load Data
print("📄 Loading cleaned dataset...")
df = pd.read_csv(CLEANED_CSV)
feature_names = ["i___beta_hcgmiu_ml", "ii____beta_hcgmiu_ml", "amhng_ml"]

# 2️⃣ Load Artifacts
print("📦 Loading pipeline and model...")
try:
    preprocessor = joblib.load(PIPELINE_FILE)
    model = joblib.load(MODEL_FILE)
except FileNotFoundError:
    print("❌ Error: Pipeline or Model file not found.")
    exit()

# 3️⃣ Create Full Pipeline (The "Magic" Step)
# This binds the preprocessor and model together so we can feed RAW data to the PDP plotter
full_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("clf", model)
])

# 4️⃣ Prepare Data (Raw Scale)
X = df[feature_names]
# Note: y is not strictly needed for PDP calculation, but good for context
print(f"Data shape: {X.shape}")

# 5️⃣ Generate 1D PDPs (One for each feature)
print("\n📊 Generating 1D Partial Dependence Plots...")
fig, ax = plt.subplots(1, 3, figsize=(18, 5))

for i, feat in enumerate(feature_names):
    print(f"   Processing feature: {feat}")
    PartialDependenceDisplay.from_estimator(
        full_pipeline,
        X=X,
        features=[feat],
        kind="average",  # Shows the average effect
        grid_resolution=50,
        ax=ax[i],
        target=1  # Focus on Class 1 (PCOS Positive)
    )
    ax[i].set_title(f"Impact of {feat} on PCOS Risk")

plt.tight_layout()
plt.savefig("pdp_1d_features.png", dpi=200)
plt.close() # Good practice to close figure
print("✅ Saved: pdp_1d_features.png")

# 6️⃣ Generate 2D PDP (Interaction between AMH and Beta HCG I)
print("\n📊 Generating 2D Interaction Plot (AMH vs Beta HCG I)...")
# This helps see if High AMH is *worse* when Beta HCG is also High
pair = ("amhng_ml", "i___beta_hcgmiu_ml")

fig2, ax2 = plt.subplots(1, 1, figsize=(7, 6))
PartialDependenceDisplay.from_estimator(
    full_pipeline,
    X=X,
    features=[pair],
    kind="average",
    grid_resolution=30, # Lower resolution for 2D to be faster
    ax=ax2,
    target=1
)
ax2.set_title(f"Interaction: {pair[0]} vs {pair[1]}")
plt.tight_layout()
plt.savefig("pdp_2d_interaction.png", dpi=200)
plt.close()
print("✅ Saved: pdp_2d_interaction.png")

# 7️⃣ Generate ICE Curves (Individual Conditional Expectation)
print("\n📊 Generating ICE Curves for AMH...")
# This shows the lines for INDIVIDUAL patients, not just the average
fig3, ax3 = plt.subplots(1, 1, figsize=(8, 6))

PartialDependenceDisplay.from_estimator(
    full_pipeline,
    X=X,
    features=["amhng_ml"],
    kind="both",          # 'both' draws the average (Orange) AND individual lines (Blue)
    grid_resolution=50,
    ax=ax3,
    target=1,
    centered=True         # Centers lines to start at 0 (easier to see trend)
)
ax3.set_title("ICE Curves: How AMH affects risk (Per Patient)")
plt.tight_layout()
plt.savefig("pdp_ice_amh.png", dpi=200)
plt.close()
print("✅ Saved: pdp_ice_amh.png")

print("\n🎉 Done! All PDP plots generated successfully.")