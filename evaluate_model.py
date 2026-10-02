import pandas as pd
import numpy as np
import joblib
from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# =========================
# Paths
# =========================
DATASET_PATH = Path("data/processed/cleaned_ddos_dataset.csv")
MODEL_PATH = Path("models/ids_model.joblib")
RESULTS_DIR = Path("results")

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# =========================
# Load Dataset
# =========================
print("Loading processed dataset...")

df = pd.read_csv(DATASET_PATH)

print(f"Dataset shape: {df.shape}")

# =========================
# Prepare Features
# =========================
X = df.drop(columns=["Label"])
y = df["Label"]

# Handle infinity values
X = X.replace([np.inf, -np.inf], np.nan)

# Fill missing values
X = X.fillna(X.median(numeric_only=True))

# Keep numeric features only
X = X.select_dtypes(include=[np.number])

print(f"Features used: {X.shape[1]}")

# =========================
# Recreate Test Split
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print(f"Test samples: {len(X_test)}")

# =========================
# Load Trained Model
# =========================
print("Loading trained model...")

model = joblib.load(MODEL_PATH)

# =========================
# Predictions
# =========================
print("Making predictions...")

y_pred = model.predict(X_test)

# =========================
# Calculate Metrics
# =========================
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

cm = confusion_matrix(y_test, y_pred)

print("\n==============================")
print("MODEL EVALUATION")
print("==============================")

print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")

print("\nConfusion Matrix:")
print(cm)

# =========================
# Save Metrics Report
# =========================
metrics = pd.DataFrame({
    "Model": ["Random Forest"],
    "Accuracy": [accuracy],
    "Precision": [precision],
    "Recall": [recall],
    "F1 Score": [f1]
})

metrics.to_csv(
    RESULTS_DIR / "metrics_report.csv",
    index=False
)

print("\nSaved: results/metrics_report.csv")

# =========================
# Confusion Matrix Plot
# =========================
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["BENIGN", "DDoS"]
)

disp.plot()

plt.title("Random Forest - Confusion Matrix")
plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "confusion_matrix.png",
    dpi=300
)

plt.close()

print("Saved: results/confusion_matrix.png")

# =========================
# Class Distribution Plot
# =========================
class_counts = y.value_counts().sort_index()

plt.figure(figsize=(7, 5))

plt.bar(
    ["BENIGN", "DDoS"],
    class_counts.values
)

plt.title("Dataset Class Distribution")
plt.xlabel("Class")
plt.ylabel("Number of Samples")

plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "class_distribution.png",
    dpi=300
)

plt.close()

print("Saved: results/class_distribution.png")

# =========================
# Model Comparison
# =========================
# These are the results obtained during training.
comparison = pd.DataFrame({
    "Model": [
        "Logistic Regression",
        "Decision Tree",
        "Random Forest"
    ],
    "Accuracy": [
        0.9988,
        0.9999,
        1.0000
    ],
    "Precision": [
        0.9988,
        0.9999,
        1.0000
    ],
    "Recall": [
        0.9990,
        0.9999,
        0.9999
    ],
    "F1 Score": [
        0.9989,
        0.9999,
        1.0000
    ]
})

plt.figure(figsize=(10, 6))

x = np.arange(len(comparison["Model"]))
width = 0.2

plt.bar(
    x - 1.5 * width,
    comparison["Accuracy"],
    width,
    label="Accuracy"
)

plt.bar(
    x - 0.5 * width,
    comparison["Precision"],
    width,
    label="Precision"
)

plt.bar(
    x + 0.5 * width,
    comparison["Recall"],
    width,
    label="Recall"
)

plt.bar(
    x + 1.5 * width,
    comparison["F1 Score"],
    width,
    label="F1 Score"
)

plt.xticks(x, comparison["Model"])
plt.ylim(0.99, 1.001)

plt.title("Model Performance Comparison")
plt.xlabel("Model")
plt.ylabel("Score")
plt.legend()

plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "model_comparison.png",
    dpi=300
)

plt.close()

print("Saved: results/model_comparison.png")

print("\n==============================")
print("EVALUATION COMPLETE")
print("==============================")