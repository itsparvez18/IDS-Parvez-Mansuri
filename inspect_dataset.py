import pandas as pd # type: ignore
from pathlib import Path

DATASET_PATH = Path("data/raw/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv")

print("=" * 60)
print("IDS DATASET INSPECTION")
print("=" * 60)

print("\nLoading dataset...")

df = pd.read_csv(DATASET_PATH)

print("\nDataset loaded successfully!")

print("\nShape:")
print(f"Rows    : {df.shape[0]:,}")
print(f"Columns : {df.shape[1]:,}")

print("\nColumn names:")
for i, column in enumerate(df.columns, 1):
    print(f"{i}. {column}")

print("\nData types:")
print(df.dtypes)

print("\nMissing values:")
missing = df.isnull().sum()
print(missing[missing > 0])

print("\nLabel distribution:")

label_column = " Label"

if label_column in df.columns:
    print(df[label_column].value_counts())
else:
    print("Label column not found.")

print("\nFirst 5 rows:")
print(df.head())

print("\n" + "=" * 60)
print("INSPECTION COMPLETE")
print("=" * 60)