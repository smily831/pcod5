# lime_explainer.py
import pandas as pd
import numpy as np
import joblib
from lime.lime_tabular import LimeTabularExplainer
import matplotlib.pyplot as plt
import random

# ---------------- CONFIG ----------------
CLEANED_CSV = "PCOS_infertility_cleaned.csv"
PIPELINE_FILE = "preprocessing_pipeline.joblib"
MODEL_FILE = "model_random_forest.joblib"
RANDOM_STATE = 42
N_SAMPLES = 5
# ----------------------------------------

# 🩺 FIX 1: Define original feature names (same as in cleaned dataset)
original_features = ["i___beta_hcgmiu_ml", "ii____beta_hcgmiu_ml", "amhng_ml"]
target_col = "pcos_y_n"

# 🩺 FIX 2: Define function used inside pipeline
def clip_nonnegative(X):
    """Ensure no negative hormone values remain."""
    X = np.array(X, dtype=float)
    X[X < 0] = 0.0
    return X

# Set seed
np.random.seed(RANDOM_STATE)
random.seed(RANDOM_STATE)

# 1️⃣ Load preprocessing pipeline and model
print("Loading preprocessing pipeline and model...")
preprocessor = joblib.load(PIPELINE_FILE)
model = joblib.load(MODEL_FILE)

# 2️⃣ Load cleaned dataset
df = pd.read_csv(CLEANED_CSV)
X = df[original_features]
y = df[target_col]

print(f"Loaded dataset with shape: {df.shape}")
print("Feature columns:", original_features)

# 3️⃣ Define model predict function for LIME (works on original data)
def model_predict_proba(X_orig_numpy):
    X_df = pd.DataFrame(X_orig_numpy, columns=original_features)
    X_trans = preprocessor.transform(X_df)
    return model.predict_proba(X_trans)

# 4️⃣ Create LIME explainer
explainer = LimeTabularExplainer(
    training_data=X.values,
    feature_names=original_features,
    class_names=["No PCOS", "PCOS"],
    mode="classification",
    discretize_continuous=False,
    random_state=RANDOM_STATE
)

# 5️⃣ Randomly select samples to explain
sample_indices = random.sample(range(len(X)), N_SAMPLES)
print(f"\nSelected test indices for explanation: {sample_indices}")

# 6️⃣ Generate explanations
for idx in sample_indices:
    x_instance = X.iloc[idx].values
    true_label = y.iloc[idx]
    pred = model_predict_proba([x_instance])[0]
    pred_class = np.argmax(pred)
    pred_prob = pred[pred_class]

    print(f"\n🔍 Explaining instance index: {idx}")
    print(f"True label: {'PCOS' if true_label == 1 else 'No PCOS'}")
    print(f"Model prediction: {'PCOS' if pred_class == 1 else 'No PCOS'} (prob: {pred_prob:.3f})")
    print(f"Feature values: {dict(zip(original_features, x_instance))}")

    exp = explainer.explain_instance(
        data_row=x_instance,
        predict_fn=model_predict_proba,
        num_features=len(original_features)
    )

    print("\nTop contributing features (LIME explanation):")
    for feat, weight in exp.as_list(label=1):
        print(f"  {feat}: {weight:.3f}")

    fig = exp.as_pyplot_figure(label=1)
    plt.title(f"LIME Explanation for Instance {idx} ({'PCOS' if pred_class == 1 else 'No PCOS'})")
    plt.tight_layout()
    plt.show()
