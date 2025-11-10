# pdp_plots.py
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.inspection import PartialDependenceDisplay

def clip_nonnegative(X):
    import numpy as _np
    X = _np.array(X, dtype=float)
    X[X < 0] = 0.0
    return X
# --------------------------------

# Filenames (adjust if you used different names)
CLEANED_CSV = "PCOS_infertility_cleaned.csv"
PIPELINE_FILE = "preprocessing_pipeline.joblib"   # your ColumnTransformer pipeline
MODEL_FILE = "model_random_forest.joblib"         # or model_logistic.joblib

# 1) Load original cleaned data (original feature scales)
df = pd.read_csv(CLEANED_CSV)
print("Loaded cleaned data shape:", df.shape)

# Feature names as in your cleaned CSV
feature_names = ["i___beta_hcgmiu_ml", "ii____beta_hcgmiu_ml", "amhng_ml"]

# 2) Load preprocessing pipeline and model
preprocessor = joblib.load(PIPELINE_FILE)
model = joblib.load(MODEL_FILE)

# 3) Compose a full pipeline: original_data -> preprocessor -> model
full_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("clf", model)
])

# 4) Prepare X (we use the original-scale feature DataFrame)
X = df[feature_names]
y = df["pcos_y_n"]

# 5) 1D PDPs for each feature (average partial dependence)
#    Using kind='average' shows the marginal effect on predicted probability.
fig, ax = plt.subplots(1, 3, figsize=(18, 4))
for i, feat in enumerate(feature_names):
    display = PartialDependenceDisplay.from_estimator(
        full_pipeline,
        X=X,
        features=[feat],
        kind="average",               # average PDP (default)
        grid_resolution=100,
        ax=ax[i],
        target=1                      # partial dependence for class 1 (PCOS=1)
    )
    ax[i].set_title(f"PDP: {feat}")
plt.tight_layout()
plt.savefig("pdp_1d_each_feature.png", dpi=200)
print("Saved: pdp_1d_each_feature.png")
plt.show()

# 6) 2D PDP for a pair (e.g., AMH vs beta1). Choose a pair that makes clinical sense.
pair = ("amhng_ml", "i___beta_hcgmiu_ml")
fig2, ax2 = plt.subplots(1, 1, figsize=(6,5))
PartialDependenceDisplay.from_estimator(
    full_pipeline,
    X=X,
    features=[pair],
    kind="average",
    grid_resolution=40,
    ax=ax2,
    target=1
)
ax2.set_title(f"2D PDP: {pair[0]} vs {pair[1]}")
plt.tight_layout()
plt.savefig("pdp_2d_amh_vs_beta1.png", dpi=200)
print("Saved: pdp_2d_amh_vs_beta1.png")
plt.show()

# 7) Optional: Individual conditional expectation (ICE) curves for a feature
#    Useful if you want to see per-instance curves instead of the averaged PDP.
#    Example for AMH:
fig3, ax3 = plt.subplots(1,1, figsize=(6,5))
PartialDependenceDisplay.from_estimator(
    full_pipeline,
    X=X,
    features=["amhng_ml"],
    kind="both",          # 'both' draws PDP (average) + ICE curves
    grid_resolution=80,
    ax=ax3,
    target=1
)
ax3.set_title("PDP + ICE: amhng_ml")
plt.tight_layout()
plt.savefig("pdp_ice_amh.png", dpi=200)
print("Saved: pdp_ice_amh.png")
plt.show()
