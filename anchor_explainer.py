# anchor_explainer.py
import pandas as pd
import numpy as np
import joblib
import random
from anchor import anchor_tabular

# ---------------- CONFIG ----------------
CLEANED_CSV = "PCOS_infertility_cleaned.csv"
PIPELINE_FILE = "preprocessing_pipeline.joblib"
MODEL_FILE = "model_random_forest.joblib"
RANDOM_STATE = 42
# ----------------------------------------

# Define feature and target columns
original_features = ["i___beta_hcgmiu_ml", "ii____beta_hcgmiu_ml", "amhng_ml"]
target_col = "pcos_y_n"

# Function used in preprocessing pipeline
def clip_nonnegative(X):
    X = np.array(X, dtype=float)
    X[X < 0] = 0.0
    return X

np.random.seed(RANDOM_STATE)
random.seed(RANDOM_STATE)

# 1️⃣ Load preprocessor and model
print("Loading preprocessing pipeline and model...")
preprocessor = joblib.load(PIPELINE_FILE)
model = joblib.load(MODEL_FILE)

# 2️⃣ Load dataset
df = pd.read_csv(CLEANED_CSV)
X = df[original_features]
y = df[target_col]
print(f"Loaded dataset with shape: {X.shape}")
print("Target value counts:\n", y.value_counts())

# 3️⃣ Preprocess for background (Anchor uses raw values though)
X_transformed = preprocessor.transform(X)

# 4️⃣ Initialize AnchorTabularExplainer
print("\nInitializing AnchorTabular explainer...")
explainer = anchor_tabular.AnchorTabularExplainer(
    class_names=["No PCOS", "PCOS"],
    feature_names=original_features,
    train_data=X.values
)

# 5️⃣ Random sample to explain
idx = random.randint(0, len(X) - 1)
instance = X.iloc[idx]
true_label = y.iloc[idx]
instance_df = pd.DataFrame([instance], columns=original_features)

pred_label = model.predict(preprocessor.transform(instance_df))[0]
pred_prob = model.predict_proba(preprocessor.transform(instance_df))[0][1]

print(f"\nExplaining instance index {idx}:")
print(f"True label: {'PCOS' if true_label == 1 else 'No PCOS'}")
print(f"Model prediction: {'PCOS' if pred_label == 1 else 'No PCOS'} (probability: {pred_prob:.3f})")
print("Feature values:")
print(instance.to_dict())

# 6️⃣ Generate explanation
print("\nGenerating Anchor explanation...")

exp = explainer.explain_instance(
    instance.values,
    model.predict,
    threshold=0.95,
    delta=0.1
)

# --- Helper function to safely extract anchor metrics ---
def safe_attr(attr):
    """Safely extract anchor precision/coverage values for all library versions"""
    if callable(attr):
        return attr()
    return attr

precision_val = safe_attr(exp.precision)
coverage_val = safe_attr(exp.coverage)

# 7️⃣ Display results
print("\n=== ANCHOR EXPLANATION ===")
print(f"Anchor Rule: {' AND '.join(exp.names())}")
print(f"Precision: {precision_val:.3f}")
print(f"Coverage: {coverage_val:.3f}")
print(f"Predicted class: {['No PCOS', 'PCOS'][int(pred_label)]}")

# Display features used in rule
if hasattr(exp, "features"):
    print(f"Rule uses features: {exp.features()}")

# Display rule text (in some versions stored in exp.anchor)
if hasattr(exp, "anchor"):
    print(f"Anchor text: {exp.anchor}")

print("\n--- Human-readable Explanation ---")
rule = "IF " + " AND ".join(exp.names()) + f" THEN prediction = {['No PCOS', 'PCOS'][int(pred_label)]}"
print(rule)
print(f"Precision: {precision_val:.3f}, Coverage: {coverage_val:.3f}")

print("\n✅ Anchor explanation process completed successfully.")
