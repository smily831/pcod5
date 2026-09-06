# anchor_explainer.py
import pandas as pd
import numpy as np
import joblib
import random
import os
# Ensure the correct library is installed: pip install anchor-exp
from anchor import anchor_tabular
from matplotlib import pyplot as plt

# ---------------- CONFIG ----------------
CLEANED_CSV = "PCOS_infertility_cleaned.csv"
PIPELINE_FILE = "preprocessing_pipeline.joblib"
MODEL_FILE = "model_random_forest.joblib"
RANDOM_STATE = 42
# ----------------------------------------

# Define feature and target columns
original_features = ["i___beta_hcgmiu_ml", "ii____beta_hcgmiu_ml", "amhng_ml"]
target_col = "pcos_y_n"


# Function used in preprocessing pipeline (Must be defined for joblib)
def clip_nonnegative(X):
    X = np.array(X, dtype=float)
    X[X < 0] = 0.0
    return X


np.random.seed(RANDOM_STATE)
random.seed(RANDOM_STATE)

# 1️⃣ Load preprocessor and model
print("Loading preprocessing pipeline and model...")
try:
    preprocessor = joblib.load(PIPELINE_FILE)
    model = joblib.load(MODEL_FILE)
except FileNotFoundError:
    print("❌ Error: Pipeline or Model file not found.")
    exit()

# 2️⃣ Load dataset
df = pd.read_csv(CLEANED_CSV)
X = df[original_features]
y = df[target_col]
print(f"Loaded dataset with shape: {X.shape}")


# 3️⃣ Define Prediction Wrapper (CRITICAL FIX)
def predict_fn(x_numpy):
    """
    Wrapper that takes RAW numpy data from Anchor,
    converts it to DataFrame, transforms it, and predicts.
    """
    # Ensure input is 2D
    if x_numpy.ndim == 1:
        x_numpy = x_numpy.reshape(1, -1)

    # Convert to DataFrame so pipeline sees column names
    x_df = pd.DataFrame(x_numpy, columns=original_features)

    # Preprocess
    x_transformed = preprocessor.transform(x_df)

    # Return predictions
    return model.predict(x_transformed)


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

# Get model prediction using our wrapper
pred_label = predict_fn(instance.values)[0]

print(f"\nExplaining instance index {idx}:")
print(f"True label: {'PCOS' if true_label == 1 else 'No PCOS'}")
print(f"Model prediction: {'PCOS' if pred_label == 1 else 'No PCOS'}")
print("Feature values:")
print(instance.to_dict())

# 6️⃣ Generate explanation
print("\nGenerating Anchor explanation...")

# Use the wrapper 'predict_fn' instead of 'model.predict'
exp = explainer.explain_instance(
    instance.values,
    predict_fn,
    threshold=0.95,
    delta=0.1
)


# --- Helper function to safely extract anchor metrics ---
def safe_attr(attr):
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

print("\n--- Human-readable Explanation ---")
rule = "IF " + " AND ".join(exp.names()) + f" THEN prediction = {['No PCOS', 'PCOS'][int(pred_label)]}"
print(rule)
print(f"Precision: {precision_val:.3f} (Rule accuracy)")
print(f"Coverage: {coverage_val:.3f} (Rule generality)")

print("\n✅ Anchor explanation process completed successfully.")

print("\n🎨 Generating Anchor 'Plot' (Text Image)...")

# 1. Create a blank white image
fig, ax = plt.figure(figsize=(10, 4)), plt.gca()
ax.axis('off')  # Turn off x/y axis numbers

# 2. Define the text to display
text_to_show = (
    f"ANCHOR EXPLANATION (Instance {idx})\n\n"
    f"Prediction: {['No PCOS', 'PCOS'][int(pred_label)]}\n\n"
    f"RULE:\n{ ' AND '.join(exp.names()) }\n\n"
    f"Metrics:\nPrecision: {precision_val:.2%}  |  Coverage: {coverage_val:.2%}"
)

# 3. Add text to the image
plt.text(
    0.5, 0.5,
    text_to_show,
    ha='center', va='center',
    fontsize=12,
    bbox=dict(boxstyle="round,pad=1", facecolor="#f0f0f0", edgecolor="black")
)

# 4. Save the image
save_path = f"anchor_explanation_{idx}.png"
plt.savefig(save_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"✅ Saved Anchor text image to: {save_path}")