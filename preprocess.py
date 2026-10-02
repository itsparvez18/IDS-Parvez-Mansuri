import pandas as pd
import numpy as np
from pathlib import Path

# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = Path(
    "data/raw/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"
)

OUTPUT_FILE = Path(
    "data/processed/cleaned_ddos_dataset.csv"
)

# ============================================================
# LOAD DATASET
# ============================================================

print("=" * 60)
print("IDS DATA PREPROCESSING")
print("=" * 60)

print("\nLoading dataset...")

df = pd.read_csv(INPUT_FILE)

print(f"Original shape: {df.shape}")

# ============================================================
# CLEAN COLUMN NAMES
# ============================================================

df.columns = df.columns.str.strip()

print("\nColumn names cleaned.")

# ============================================================
# REMOVE UNNECESSARY IDENTIFIER COLUMNS
# ============================================================

columns_to_drop = [
    "Flow ID",
    "Source IP",
    "Destination IP",
    "Timestamp"
]

existing_columns = [
    column for column in columns_to_drop
    if column in df.columns
]

df = df.drop(columns=existing_columns)

print("\nRemoved identifier columns:")
for column in existing_columns:
    print(f"- {column}")

# ============================================================
# CLEAN NUMERIC VALUES
# ============================================================

print("\nCleaning numeric values...")

numeric_columns = df.select_dtypes(
    include=[np.number]
).columns

# Replace infinite values with NaN
df[numeric_columns] = df[numeric_columns].replace(
    [np.inf, -np.inf],
    np.nan
)

# Fill missing numeric values using median
for column in numeric_columns:
    if df[column].isnull().any():
        median_value = df[column].median()
        df[column] = df[column].fillna(median_value)

# ============================================================
# REMOVE DUPLICATES
# ============================================================

before_duplicates = len(df)

df = df.drop_duplicates()

after_duplicates = len(df)

print(
    f"\nDuplicate rows removed: "
    f"{before_duplicates - after_duplicates:,}"
)

# ============================================================
# CLEAN LABEL
# ============================================================

if "Label" not in df.columns:
    raise ValueError("Label column not found!")

df["Label"] = df["Label"].str.strip()

print("\nLabels:")
print(df["Label"].value_counts())

# ============================================================
# CONVERT LABEL TO BINARY
# ============================================================

df["Label"] = df["Label"].map({
    "BENIGN": 0,
    "DDoS": 1
})

# Remove rows with unexpected labels
df = df.dropna(subset=["Label"])

df["Label"] = df["Label"].astype(int)

# ============================================================
# SAVE CLEANED DATASET
# ============================================================

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

df.to_csv(
    OUTPUT_FILE,
    index=False
)

# ============================================================
# FINAL INFORMATION
# ============================================================

print("\nFinal dataset shape:")
print(df.shape)

print("\nFinal label distribution:")
print(df["Label"].value_counts())

print("\nSaved cleaned dataset to:")
print(OUTPUT_FILE)

print("\n" + "=" * 60)
print("PREPROCESSING COMPLETE")
print("=" * 60)