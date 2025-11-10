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
X_TRAIN_CSV = "X_train.csv"
X_TEST_CSV = "X_test.csv"
Y_TRAIN_CSV = "y_train.csv"
Y_TEST_CSV = "y_test.csv"

# 1. Load cleaned CSV
df = pd.read_csv(CLEANED_CSV)
print("Loaded shape:", df.shape)

# 2. Column names (based on your cleaned file)
target_col = "pcos_y_n"
beta1 = "i___beta_hcgmiu_ml"
beta2 = "ii____beta_hcgmiu_ml"
amh = "amhng_ml"

feature_cols = [beta1, beta2, amh]

# 3. Sanity checks
if target_col not in df.columns:
    raise RuntimeError(f"Target column {target_col} not found in {CLEANED_CSV}. Columns: {df.columns.tolist()}")

print("\nTarget distribution:")
print(df[target_col].value_counts())

# 4. Define transformers
# - For beta columns: clip negatives to 0 (just in case), then log1p, then scale
# - For beta columns: clip negatives to 0 (just in case), then log1p, then scale
def clip_nonnegative(X):
    X = np.array(X, dtype=float)
    X[X < 0] = 0.0
    return X

clip_transformer = FunctionTransformer(clip_nonnegative, validate=False)
log1p_transformer = FunctionTransformer(np.log1p, validate=False)

beta_pipeline = Pipeline([
    ("clip", clip_transformer),
    ("log1p", log1p_transformer),
    ("scaler", StandardScaler())
])


# - For AMH: impute median then scale
amh_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# 5. ColumnTransformer to apply appropriate pipeline to each column
preprocessor = ColumnTransformer([
    ("beta1", beta_pipeline, [beta1]),
    ("beta2", beta_pipeline, [beta2]),
    ("amh", amh_pipeline, [amh])
], remainder="drop", verbose_feature_names_out=False)

# 6. Split X and y (stratified)
X = df[feature_cols].copy()
y = df[target_col].copy()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)

print("\nShapes after split:")
print("X_train:", X_train.shape, "X_test:", X_test.shape)
print("y_train distribution:\n", y_train.value_counts())
print("y_test distribution:\n", y_test.value_counts())

# 7. Fit the preprocessor on training data and transform
preprocessor.fit(X_train)
X_train_transformed = preprocessor.transform(X_train)
X_test_transformed = preprocessor.transform(X_test)

print("\nTransformed shapes (numpy arrays):",
      X_train_transformed.shape, X_test_transformed.shape)

# 8. Save the fitted pipeline (so you can apply same preprocessing during inference)
joblib.dump(preprocessor, PIPELINE_FILE)
print(f"\nSaved preprocessing pipeline to: {PIPELINE_FILE}")

# 9. Save the transformed datasets as CSV (with same column names)
# Recreate DataFrames from transformed arrays with descriptive column names
transformed_feature_names = ["beta1_log_scaled", "beta2_log_scaled", "amh_scaled"]
X_train_df = pd.DataFrame(X_train_transformed, columns=transformed_feature_names, index=X_train.index)
X_test_df = pd.DataFrame(X_test_transformed, columns=transformed_feature_names, index=X_test.index)

X_train_df.to_csv(X_TRAIN_CSV, index=False)
X_test_df.to_csv(X_TEST_CSV, index=False)
y_train.to_csv(Y_TRAIN_CSV, index=False)
y_test.to_csv(Y_TEST_CSV, index=False)

print(f"Saved: {X_TRAIN_CSV}, {X_TEST_CSV}, {Y_TRAIN_CSV}, {Y_TEST_CSV}")

# 10. Short example: how to load pipeline and transform a new sample
def transform_new_sample(sample_dict, pipeline_path=PIPELINE_FILE):
    """
    sample_dict must have keys: beta1, beta2, amh (use original column names)
    e.g. {"i___beta_hcgmiu_ml": 20.0, "ii____beta_hcgmiu_ml": 1.99, "amhng_ml": 3.5}
    """
    pipe = joblib.load(pipeline_path)
    sample_df = pd.DataFrame([sample_dict])[feature_cols]
    transformed = pipe.transform(sample_df)
    return pd.DataFrame(transformed, columns=transformed_feature_names)

# Example usage:
example = {
    beta1: 20.0,
    beta2: 1.99,
    amh: 3.5
}
print("\nExample transformed sample:")
print(transform_new_sample(example))

