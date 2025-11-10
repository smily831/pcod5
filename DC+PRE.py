# data_cleaning_preprocessing.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. Load
df = pd.read_csv("PCOS_infertility.csv")

# 2. Clean column names (strip spaces, lower-case, replace special chars)
df.columns = [c.strip().lower().replace(" ", "_").replace("/", "_").replace("(", "").replace(")", "").replace("-", "_") for c in df.columns]

# 3. Inspect current columns
print("Columns after rename:", df.columns.tolist())

# 4. Convert AMH column to numeric (coerce errors to NaN), identify problematic entries
amh_col = None
for c in df.columns:
    if "amh" in c:
        amh_col = c
        break

if amh_col is None:
    raise RuntimeError("Could not find AMH column after renaming. Column names: " + str(df.columns.tolist()))

# Show unique non-numeric-looking values (helpful for debugging)
non_numeric_mask = ~df[amh_col].astype(str).str.replace(".", "", 1).str.isnumeric()
print("Sample non-numeric AMH values (up to 20):")
print(df.loc[non_numeric_mask, amh_col].unique()[:20])

# Convert to numeric
df[amh_col] = pd.to_numeric(df[amh_col], errors="coerce")

# 5. Drop ID-like columns that won't be used for modeling
drop_cols = []
for c in df.columns:
    if c.startswith("sl") or "patient" in c or "file" in c:
        drop_cols.append(c)

df = df.drop(columns=drop_cols, errors="ignore")
print(f"Dropped columns: {drop_cols}")

# 6. Check missing values after conversion
print("\nMissing values per column after AMH conversion:")
print(df.isnull().sum())

# 7. Target column: make sure it's named and is integer 0/1
target_col = None
for c in df.columns:
    if "pcos" in c:
        target_col = c
        break

if target_col is None:
    raise RuntimeError("Could not find target column (PCOS). Columns: " + str(df.columns.tolist()))

# If it's not 0/1, try to map common representations
print("\nTarget value counts before any mapping:")
print(df[target_col].value_counts(dropna=False))

# If target is string like 'Y'/'N' convert; else assume numeric 0/1
if df[target_col].dtype == object:
    df[target_col] = df[target_col].str.strip().str.upper().map({"Y":1, "YES":1, "N":0, "NO":0}).astype("Int64")

# Ensure integer dtype
df[target_col] = df[target_col].astype("Int64")

print("\nTarget value counts after processing:")
print(df[target_col].value_counts(dropna=False))

# 8. Basic statistics of numeric features
print("\nNumeric summary:")
print(df.describe(include=[np.number]).T)

# 9. Simple class-balance bar plot
plt.figure(figsize=(5,4))
counts = df[target_col].value_counts().sort_index()
labels = ["No PCOS (0)", "PCOS (1)"] if set(counts.index).issuperset({0,1}) else counts.index.astype(str)
plt.bar(labels, counts.values)
plt.title("Target class distribution")
plt.ylabel("Count")
for i, v in enumerate(counts.values):
    plt.text(i, v+2, str(v), ha="center")
plt.tight_layout()
plt.show()

# 10. Save cleaned CSV for later steps
df.to_csv("PCOS_infertility_cleaned.csv", index=False)
print("\nCleaned dataset saved to PCOS_infertility_cleaned.csv")
