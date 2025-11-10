# ice_explainer.py
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.inspection import PartialDependenceDisplay

# ---------- CONFIGURATION ----------
CLEANED_CSV = "PCOS_infertility_cleaned.csv"
PIPELINE_FILE = "preprocessing_pipeline.joblib"
MODEL_FILE = "model_random_forest.joblib"
FEATURES = ["i___beta_hcgmiu_ml", "ii____beta_hcgmiu_ml", "amhng_ml"]
TARGET = "pcos_y_n"
RANDOM_STATE = 42

# Function used in the preprocessing pipeline
def clip_nonnegative(X):
    X = np.array(X, dtype=float)
    X[X < 0] = 0.0
    return X

# ---------- STEP 1: Load pipeline, model, data ----------
print("Loading preprocessing pipeline and model...")
preprocessor = joblib.load(PIPELINE_FILE)
model = joblib.load(MODEL_FILE)
df = pd.read_csv(CLEANED_CSV)
print(f"Dataset loaded: {df.shape[0]} samples, {df.shape[1]} columns")

# ---------- STEP 2: Prepare data ----------
X = df[FEATURES]
y = df[TARGET]
X_preprocessed = preprocessor.transform(X)
print(f"Preprocessed X shape: {X_preprocessed.shape}")

# ---------- STEP 3: Choose feature(s) for ICE ----------
feature_to_explain = "amhng_ml"  # you can change to any feature
feat_idx = FEATURES.index(feature_to_explain)
print(f"Explaining feature: {feature_to_explain} (index {feat_idx})")

# ---------- STEP 4: Generate PDP + ICE ----------
fig, ax = plt.subplots(figsize=(8, 5))
PartialDependenceDisplay.from_estimator(
    estimator=model,
    X=X_preprocessed,
    features=[feat_idx],
    kind="both",  # <- BOTH = PDP + ICE together
    grid_resolution=60,
    ax=ax,
)
ax.set_title(f"ICE + PDP for {feature_to_explain}")
ax.set_ylabel("Predicted probability (PCOS=1)")
plt.tight_layout()
plt.show()

# ---------- STEP 5: Optional 2D ICE (interaction) ----------
fig2, ax2 = plt.subplots(figsize=(6, 5))
PartialDependenceDisplay.from_estimator(
    estimator=model,
    X=X_preprocessed,
    features=[(FEATURES.index("amhng_ml"), FEATURES.index("i___beta_hcgmiu_ml"))],
    kind="average",
    grid_resolution=40,
    ax=ax2,
)
ax2.set_title("2D PDP: AMH vs I β-hCG (Interaction)")
plt.tight_layout()
plt.show()
