
from pathlib import Path

import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_recall_curve,
    classification_report,
    average_precision_score
)

ROOT = Path(__file__).resolve().parent

DATA_PATH = (
    ROOT / "data" /
    "PS_20174392719_1491204439457_log.csv"
)
MODEL_PATH = ROOT / "models" / "random_forest.joblib"
REPORT_DIR = ROOT / "reports"

REPORT_DIR.mkdir(parents=True, exist_ok=True)

# 1. Load the saved model and threshold
bundle = joblib.load(MODEL_PATH)
model = bundle["pipeline"]
threshold = bundle["threshold"]
features = bundle["features"]

print(f"Loaded model: {bundle['model_name']}")
print(f"Decision threshold: {threshold:.6f}")

# 2. Reproduce the test split used in train.py
df = pd.read_csv(DATA_PATH)

required = features + ["isFraud"]
df = df.dropna(subset=required).copy()
df["isFraud"] = df["isFraud"].astype(int)

max_step = df["step"].max()
test_start = int(max_step * 0.85)

test = df[df["step"] > test_start].copy()

X_test = test[features]
y_test = test["isFraud"]

print(f"Test transactions: {len(test):,}")
print(f"Test fraud cases: {int(y_test.sum()):,}")

# 3. Generate predictions
probabilities = model.predict_proba(X_test)[:, 1]
predictions = (probabilities >= threshold).astype(int)

print("\nClassification report:")
print(
    classification_report(
        y_test,
        predictions,
        target_names=["Legitimate", "Fraud"],
        zero_division=0
    )
)

print(
    f"Average precision: "
    f"{average_precision_score(y_test, probabilities):.4f}"
)

# 4. Confusion matrix
cm = confusion_matrix(
    y_test,
    predictions,
    labels=[0, 1]
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Legitimate", "Fraud"]
)

disp.plot(values_format=",")
plt.title("Random Forest - Confusion Matrix")
plt.tight_layout()

cm_path = REPORT_DIR / "confusion_matrix.png"
plt.savefig(cm_path, dpi=200, bbox_inches="tight")
plt.show()

print(f"Saved: {cm_path}")

# 5. Precision-recall curve
precision, recall, thresholds = precision_recall_curve(
    y_test,
    probabilities
)

ap = average_precision_score(y_test, probabilities)

plt.figure(figsize=(8, 6))
plt.plot(recall, precision, label=f"Random Forest (AP={ap:.3f})")
plt.axvline(
    recall_score_value := cm[1, 1] / max(cm[1, 1] + cm[1, 0], 1),
    linestyle="--",
    label=f"Selected-threshold recall={recall_score_value:.3f}"
)

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

pr_path = REPORT_DIR / "precision_recall_curve.png"
plt.savefig(pr_path, dpi=200, bbox_inches="tight")
plt.show()

print(f"Saved: {pr_path}")
print("\nEvaluation complete.")
