# preprocess_and_split.py
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import os

RANDOM_STATE = 42
TEST_SIZE = 0.2
CLEANED_CSV = "PCOS_infertility_cleaned.csv"
PIPELINE_FILE = "preprocessing_pipeline.joblib"
# We save these as RAW data so XAI tools can read them easily
X_TRAIN_CSV = "X_train.csv"
X_TEST_CSV = "X_test.csv"
Y_TRAIN_CSV = "y_train.csv"
Y_TEST_CSV = "y_test.csv"

# 1. Load cleaned CSV
print("📦 Loading dataset...")
df = pd.read_csv(CLEANED_CSV)

# 2. Define Features
target_col = "pcos_y_n"
beta1 = "i___beta_hcgmiu_ml"
beta2 = "ii____beta_hcgmiu_ml"
amh = "amhng_ml"
feature_cols = [beta1, beta2, amh]

# 3. Define Transformations
def clip_nonnegative(X):
    X = np.array(X, dtype=float)
    X[X < 0] = 0.0
    return X

clip_transformer = FunctionTransformer(clip_nonnegative, validate=False)
log1p_transformer = FunctionTransformer(np.log1p, validate=False)

# Pipeline for skewed features (Beta HCG)
beta_pipeline = Pipeline([
    ("clip", clip_transformer),
    ("log1p", log1p_transformer),
    ("scaler", StandardScaler())
])

# Pipeline for normal-ish features (AMH)
amh_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# Full Preprocessor
preprocessor = ColumnTransformer([
    ("beta1", beta_pipeline, [beta1]),
    ("beta2", beta_pipeline, [beta2]),
    ("amh", amh_pipeline, [amh])
], remainder="drop", verbose_feature_names_out=False)

# 4. Split Data (Stratified)
X = df[feature_cols].copy()
y = df[target_col].copy()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)

print(f"✅ Split successful: Train {X_train.shape}, Test {X_test.shape}")

# 5. Fit Pipeline on Train Data Only (Prevent Leakage)
print("⚙️ Fitting preprocessing pipeline...")
preprocessor.fit(X_train)

# 6. Save Pipeline
joblib.dump(preprocessor, PIPELINE_FILE)
print(f"💾 Saved pipeline to: {PIPELINE_FILE}")

# 7. Save RAW Datasets (CRITICAL CHANGE)
# We save the raw values so your XAI scripts (PDP, LIME) can show real units.
X_train.to_csv(X_TRAIN_CSV, index=False)
X_test.to_csv(X_TEST_CSV, index=False)
y_train.to_csv(Y_TRAIN_CSV, index=False)
y_test.to_csv(Y_TEST_CSV, index=False)

print(f"💾 Saved RAW split datasets: {X_TRAIN_CSV}, {X_TEST_CSV}")
print("   (These contain original units like ng/ml, perfect for XAI)")