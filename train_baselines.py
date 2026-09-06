import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer  # <--- FIXED: Needed to handle NaNs
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)
import os
import sys

# --- Configuration ---
RANDOM_STATE = 42
X_TRAIN_CSV = "X_train.csv"
X_TEST_CSV = "X_test.csv"
Y_TRAIN_CSV = "y_train.csv"
Y_TEST_CSV = "y_test.csv"
FEATURE_NAMES_FILE = 'feature_names.joblib'
IMPUTER_FILE = 'imputer.joblib'  # <--- New file to save the "filler" logic

# 1. Load Data
print("📦 Loading datasets...")
if not os.path.exists(X_TRAIN_CSV):
    print(f"❌ Error: {X_TRAIN_CSV} not found.")
    sys.exit(1)

X_train = pd.read_csv(X_TRAIN_CSV)
X_test = pd.read_csv(X_TEST_CSV)
y_train = pd.read_csv(Y_TRAIN_CSV).squeeze()
y_test = pd.read_csv(Y_TEST_CSV).squeeze()

X_train_np = X_train.values
X_test_np = X_test.values

# --- 2. FIX: Handle Missing Values (Imputation) ---
print("🔧 Imputing missing values...")
# Create an imputer that fills blanks with the average (mean)
imputer = SimpleImputer(strategy='mean')

# Learn the averages from Train data and fill blanks
X_train_np = imputer.fit_transform(X_train_np)

# Fill blanks in Test data using the SAME averages (do not fit again!)
X_test_np = imputer.transform(X_test_np)

# Save the imputer so we can use it in the App/XAI later
joblib.dump(imputer, IMPUTER_FILE)
print(f"✅ Missing values filled & Imputer saved to: {IMPUTER_FILE}")

# Save feature names once (shared by both models)
feature_names = X_train.columns.tolist()
joblib.dump(feature_names, FEATURE_NAMES_FILE)
print(f"✅ Saved feature names to: {FEATURE_NAMES_FILE}")

# 3. Define Models
models = {
    "LogisticRegression": LogisticRegression(
        random_state=RANDOM_STATE,
        solver='liblinear',
        class_weight='balanced'
    ),
    "RandomForestClassifier": RandomForestClassifier(
        random_state=RANDOM_STATE,
        n_estimators=100,
        max_depth=5,
        class_weight='balanced'
    )
}

print("\n--- Model Training & Evaluation ---")

# 4. Loop through both models
for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train_np, y_train)

    # Predictions
    y_pred = model.predict(X_test_np)
    y_prob = model.predict_proba(X_test_np)[:, 1]

    # --- Metrics ---
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print(f"  > Accuracy : {accuracy:.4f}")
    print(f"  > Precision: {precision:.4f}")
    print(f"  > Recall   : {recall:.4f}")
    print(f"  > F1 Score : {f1:.4f}")
    print(f"  > ROC-AUC  : {auc:.4f}")

    print("\n  > Confusion Matrix:")
    print(cm)

    print("\n  > Full Classification Report:")
    print(report)

    # 💾 SAVE EACH MODEL
    filename = f"model_{name}.joblib"
    joblib.dump(model, filename)
    print(f"  ✅ Saved model to: {filename}")

print("\n🎉 Done! All models and tools are saved.")