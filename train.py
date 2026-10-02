import pandas as pd
import numpy as np
import joblib

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_PATH = Path(
    "data/processed/cleaned_ddos_dataset.csv"
)

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOAD DATASET
# ============================================================

print("=" * 70)
print("AI-POWERED NETWORK INTRUSION DETECTION SYSTEM")
print("MODEL TRAINING")
print("=" * 70)

print("\nLoading processed dataset...")

df = pd.read_csv(DATASET_PATH)

print(f"Dataset shape: {df.shape}")


# ============================================================
# SEPARATE FEATURES AND TARGET
# ============================================================

X = df.drop(columns=["Label"])
y = df["Label"]

print(f"\nFeatures: {X.shape[1]}")
print(f"Samples : {X.shape[0]}")


# ============================================================
# HANDLE NUMERIC VALUES
# ============================================================

X = X.replace([np.inf, -np.inf], np.nan)

X = X.fillna(X.median(numeric_only=True))

# Make sure every feature is numeric
X = X.select_dtypes(include=[np.number])


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTrain/Test split:")
print(f"Training samples: {len(X_train):,}")
print(f"Testing samples : {len(X_test):,}")


# ============================================================
# DEFINE MODELS
# ============================================================

models = {

    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000,
                random_state=42
            )
        )
    ]),

    "Decision Tree": DecisionTreeClassifier(
        random_state=42,
        max_depth=20
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        random_state=42,
        n_jobs=-1
    )
}


# ============================================================
# TRAIN AND EVALUATE
# ============================================================

results = {}

for name, model in models.items():

    print("\n" + "=" * 70)
    print(f"TRAINING: {name}")
    print("=" * 70)

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)

    results[name] = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

    print("\nPerformance:")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            predictions,
            target_names=["BENIGN", "DDoS"]
        )
    )

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, predictions))


# ============================================================
# FIND BEST MODEL BY F1 SCORE
# ============================================================

best_model_name = max(
    results,
    key=lambda model_name: results[model_name]["f1"]
)

best_model = models[best_model_name]

print("\n" + "=" * 70)
print("MODEL COMPARISON")
print("=" * 70)

for name, metrics in results.items():

    print(
        f"\n{name}"
        f"\n  Accuracy : {metrics['accuracy']:.4f}"
        f"\n  Precision: {metrics['precision']:.4f}"
        f"\n  Recall   : {metrics['recall']:.4f}"
        f"\n  F1 Score : {metrics['f1']:.4f}"
    )

print("\n" + "=" * 70)
print(f"SELECTED MODEL: {best_model_name}")
print("=" * 70)


# ============================================================
# SAVE BEST MODEL
# ============================================================

model_path = MODEL_DIR / "ids_model.joblib"

joblib.dump(
    best_model,
    model_path
)

print(f"\nModel saved to:")
print(model_path)

print("\n" + "=" * 70)
print("MODEL TRAINING COMPLETE")
print("=" * 70)