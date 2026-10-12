
from pathlib import Path
import json
import joblib
import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve
)

# ==================================================
# 1. PROJECT PATHS AND CONFIGURATION
# ==================================================

ROOT = Path(__file__).resolve().parent

DATA_PATH = (
    ROOT / "data" /
    "PS_20174392719_1491204439457_log.csv"
)

MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
NEGATIVE_TO_FRAUD_RATIO = 10

FEATURES = [
    "step",
    "type",
    "amount",
    "oldbalanceOrg",
    "oldbalanceDest"
]

TARGET = "isFraud"

# ==================================================
# 2. LOAD AND VALIDATE THE DATASET
# ==================================================

print("=" * 60)
print("FRAUD DETECTION SYSTEM")
print("=" * 60)

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found at: {DATA_PATH}\n"
        "Check the CSV filename and data folder."
    )

print("\nLoading dataset...")
df = pd.read_csv(DATA_PATH)

required_columns = FEATURES + [TARGET]

missing_columns = set(required_columns) - set(df.columns)

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )

# Remove rows missing required values.
# Do not remove duplicates automatically: separate
# transactions can legitimately have identical values.
df = df.dropna(subset=required_columns).copy()

df[TARGET] = df[TARGET].astype(int)

if not set(df[TARGET].unique()).issubset({0, 1}):
    raise ValueError("isFraud must contain only 0 and 1.")

if not df[TARGET].isin([0, 1]).all():
    raise ValueError("Invalid target labels.")

if (df["amount"] < 0).any():
    raise ValueError("Negative transaction amounts found.")

print(f"Rows available: {len(df):,}")
print(f"Fraud cases: {(df[TARGET] == 1).sum():,}")
print(f"Legitimate cases: {(df[TARGET] == 0).sum():,}")
print(f"Fraud rate: {df[TARGET].mean() * 100:.4f}%")

# ==================================================
# 3. CHRONOLOGICAL TRAIN / VALIDATION / TEST SPLIT
# ==================================================

# Earlier transactions train the model.
# Later transactions validate and test it.

max_step = df["step"].max()

train_cutoff = int(max_step * 0.70)
validation_cutoff = int(max_step * 0.85)

train = df[df["step"] <= train_cutoff].copy()

validation = df[
    (df["step"] > train_cutoff) &
    (df["step"] <= validation_cutoff)
].copy()

test = df[df["step"] > validation_cutoff].copy()

if min(len(train), len(validation), len(test)) == 0:
    raise ValueError("One of the data splits is empty.")

for name, split in [
    ("Training", train),
    ("Validation", validation),
    ("Test", test)
]:
    fraud_count = int(split[TARGET].sum())

    print(
        f"{name}: {len(split):,} rows | "
        f"{fraud_count:,} fraud cases | "
        f"{split[TARGET].mean() * 100:.4f}% fraud"
    )

if validation[TARGET].nunique() < 2:
    raise ValueError(
        "Validation split needs both fraud and "
        "legitimate transactions."
    )

if test[TARGET].nunique() < 2:
    raise ValueError(
        "Test split needs both fraud and "
        "legitimate transactions."
    )

# ==================================================
# 4. UNDERSAMPLE TRAINING DATA ONLY
# ==================================================

fraud_train = train[train[TARGET] == 1]
legitimate_train = train[train[TARGET] == 0]

if fraud_train.empty:
    raise ValueError("No fraud cases in training data.")

sample_size = min(
    len(legitimate_train),
    len(fraud_train) * NEGATIVE_TO_FRAUD_RATIO
)

legitimate_sample = legitimate_train.sample(
    n=sample_size,
    random_state=RANDOM_STATE
)

balanced_train = pd.concat(
    [fraud_train, legitimate_sample],
    ignore_index=True
)

balanced_train = balanced_train.sample(
    frac=1,
    random_state=RANDOM_STATE
).reset_index(drop=True)

print("\nAfter training-set undersampling:")
print(f"Training rows: {len(balanced_train):,}")
print(f"Fraud cases: {balanced_train[TARGET].sum():,}")
print(
    f"Fraud rate: "
    f"{balanced_train[TARGET].mean() * 100:.2f}%"
)

# Validation and test sets remain unchanged.

X_train = balanced_train[FEATURES]
y_train = balanced_train[TARGET]

X_val = validation[FEATURES]
y_val = validation[TARGET]

X_test = test[FEATURES]
y_test = test[TARGET]

# ==================================================
# 5. PREPROCESSING
# ==================================================

numeric_features = [
    "step",
    "amount",
    "oldbalanceOrg",
    "oldbalanceDest"
]

categorical_features = ["type"]


def make_preprocessor():
    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                StandardScaler(),
                numeric_features
            ),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_features
            )
        ]
    )


# ==================================================
# 6. DEFINE THE MODELS
# ==================================================

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=500,
        random_state=RANDOM_STATE
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=RANDOM_STATE
    )
}


# ==================================================
# 7. THRESHOLD SELECTION USING VALIDATION DATA
# ==================================================


def choose_threshold(y_true, probabilities):
    """
    Target at least 95% recall on validation data.
    Among eligible thresholds, choose the highest
    threshold to preserve as much precision as possible.
    """

    target_recall = 0.95

    precision, recall, thresholds = (
        precision_recall_curve(y_true, probabilities)
    )

    if len(thresholds) == 0:
        return 0.5

    # Each threshold corresponds to one precision/recall
    # pair, excluding the final point in each array.
    precision = precision[:-1]
    recall = recall[:-1]

    eligible = np.where(recall >= target_recall)[0]

    if len(eligible) == 0:
        # Fallback: choose the threshold with maximum recall.
        return float(thresholds[np.argmax(recall)])

    # Thresholds are sorted in ascending order.
    # Choose the highest one meeting the recall target.
    best_index = eligible[-1]

    print(
        f"Validation precision at selected threshold: "
        f"{precision[best_index]:.4f}"
    )
    print(
        f"Validation recall at selected threshold: "
        f"{recall[best_index]:.4f}"
    )

    return float(thresholds[best_index])

# ==================================================
# 8. EVALUATION
# ==================================================

def evaluate_model(y_true, probabilities, threshold):
    predictions = (
        probabilities >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1]
    ).ravel()

    return {
        "threshold": float(threshold),
        "precision": float(
            precision_score(
                y_true, predictions, zero_division=0
            )
        ),
        "recall": float(
            recall_score(
                y_true, predictions, zero_division=0
            )
        ),
        "f1_score": float(
            f1_score(
                y_true, predictions, zero_division=0
            )
        ),
        "roc_auc": float(
            roc_auc_score(y_true, probabilities)
        ),
        "average_precision": float(
            average_precision_score(
                y_true, probabilities
            )
        ),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp)
    }


# ==================================================
# 9. TRAIN, VALIDATE AND TEST
# ==================================================

results = {}

for name, estimator in models.items():

    print("\n" + "=" * 60)
    print(f"Training {name}")
    print("=" * 60)

    pipeline = Pipeline(
        steps=[
            ("preprocessor", make_preprocessor()),
            ("classifier", estimator)
        ]
    )

    # Fit only on the undersampled training set.
    pipeline.fit(X_train, y_train)

    # Choose threshold using validation data only.
    validation_probabilities = (
        pipeline.predict_proba(X_val)[:, 1]
    )

    threshold = choose_threshold(
        y_val,
        validation_probabilities
    )

    # Final evaluation on untouched test data.
    test_probabilities = (
        pipeline.predict_proba(X_test)[:, 1]
    )

    metrics = evaluate_model(
        y_test,
        test_probabilities,
        threshold
    )

    results[name] = metrics

    print(f"Threshold: {threshold:.6f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1-score: {metrics['f1_score']:.4f}")
    print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
    print(
        f"Average precision: "
        f"{metrics['average_precision']:.4f}"
    )

    print("\nConfusion matrix counts:")
    print(f"True negatives: {metrics['true_negatives']:,}")
    print(f"False positives: {metrics['false_positives']:,}")
    print(f"False negatives: {metrics['false_negatives']:,}")
    print(f"True positives: {metrics['true_positives']:,}")

    # Save the pipeline and its decision threshold together.
    model_path = (
        MODEL_DIR /
        (
            "logistic_regression.joblib"
            if name == "Logistic Regression"
            else "random_forest.joblib"
        )
    )

    joblib.dump(
        {
            "pipeline": pipeline,
            "threshold": threshold,
            "features": FEATURES,
            "model_name": name
        },
        model_path
    )

    print(f"\nSaved model: {model_path}")


# ==================================================
# 10. SAVE METRICS AND COMPARISON
# ==================================================

with open(
    REPORT_DIR / "metrics.json",
    "w",
    encoding="utf-8"
) as file:
    json.dump(results, file, indent=4)

comparison = pd.DataFrame(results).T
comparison.to_csv(
    REPORT_DIR / "model_comparison.csv",
    index=True,
    index_label="model"
)

print("\n" + "=" * 60)
print("FINAL MODEL COMPARISON")
print("=" * 60)

print(
    comparison[
        [
            "precision",
            "recall",
            "f1_score",
            "roc_auc",
            "average_precision"
        ]
    ].to_string()
)

print("\nTraining completed.")
print(f"Models saved in: {MODEL_DIR}")
print(f"Reports saved in: {REPORT_DIR}")
