# lime_explainer_fixed.py
import pandas as pd
import numpy as np
import joblib
from lime.lime_tabular import LimeTabularExplainer
import matplotlib.pyplot as plt
import random
import os

# ---------------- CONFIG ----------------
CLEANED_CSV = "PCOS_infertility_cleaned.csv"
PIPELINE_FILE = "preprocessing_pipeline.joblib"
MODEL_FILE = "model_random_forest.joblib"
OUT_DIR = "lime_outputs"
RANDOM_STATE = 42
N_SAMPLES = 5
# ----------------------------------------

# Define feature names explicitly
original_features = ["i___beta_hcgmiu_ml", "ii____beta_hcgmiu_ml", "amhng_ml"]
target_col = "pcos_y_n"


# Helper function for pipeline
def clip_nonnegative(X):
    X = np.array(X, dtype=float)
    X[X < 0] = 0.0
    return X


# Setup environment
np.random.seed(RANDOM_STATE)
random.seed(RANDOM_STATE)
os.makedirs(OUT_DIR, exist_ok=True)

# 1️⃣ Load pipeline and model
print("📦 Loading preprocessing pipeline and model...")
try:
    preprocessor = joblib.load(PIPELINE_FILE)
    model = joblib.load(MODEL_FILE)
except FileNotFoundError:
    print("❌ Error: Pipeline or Model file not found.")
    exit()

# 2️⃣ Load data
print("📄 Loading dataset...")
df = pd.read_csv(CLEANED_CSV)
X = df[original_features]
y = df[target_col]

# Check for constant columns (The likely cause of your crash)
stds = X.std()
print("\n📊 Feature Standard Deviations (Low std causes LIME errors):")
print(stds)
if any(stds < 0.001):
    print("⚠️ WARNING: Some features have near-zero variance. This causes LIME crashes.")


# 3️⃣ Prediction Wrapper
def model_predict_proba(X_orig_numpy):
    # LIME generates numpy arrays, but our pipeline expects a DataFrame with headers
    X_df = pd.DataFrame(X_orig_numpy, columns=original_features)
    X_trans = preprocessor.transform(X_df)
    return model.predict_proba(X_trans)


# 4️⃣ Initialize LIME Explainer
print("\n🧠 Initializing LIME Explainer...")
explainer = LimeTabularExplainer(
    training_data=X.values,
    feature_names=original_features,
    class_names=["No PCOS", "PCOS"],
    mode="classification",
    # 🔴 FIX: Set this to False to prevent 'scale must be positive' errors
    discretize_continuous=False,
    random_state=RANDOM_STATE
)

# 5️⃣ Pick random samples
sample_indices = random.sample(range(len(X)), N_SAMPLES)
print(f"🔍 Selected indices to explain: {sample_indices}")

# 6️⃣ Generate and Save Explanations
for idx in sample_indices:
    x_instance = X.iloc[idx].values
    true_label = y.iloc[idx]

    # Get model prediction
    pred_probs = model_predict_proba([x_instance])[0]
    pred_class = np.argmax(pred_probs)

    print(f"\nProcessing Index {idx}...")
    print(f"  True Label: {'PCOS' if true_label == 1 else 'No PCOS'}")
    print(f"  Prediction: {'PCOS' if pred_class == 1 else 'No PCOS'} (Prob: {pred_probs[1]:.2f})")

    # Generate explanation
    exp = explainer.explain_instance(
        data_row=x_instance,
        predict_fn=model_predict_proba,
        num_features=len(original_features),
        num_samples=5000
    )

    # Save Plot
    fig = exp.as_pyplot_figure(label=1)
    plt.title(f"LIME: Patient {idx} ({'PCOS' if pred_class == 1 else 'No PCOS'})")
    plt.tight_layout()

    save_path = os.path.join(OUT_DIR, f"lime_patient_{idx}.png")
    plt.savefig(save_path)
    plt.close()
    print(f"  ✅ Saved plot to {save_path}")

print(f"\n🎉 Done! All {N_SAMPLES} LIME explanations are in the '{OUT_DIR}' folder.")