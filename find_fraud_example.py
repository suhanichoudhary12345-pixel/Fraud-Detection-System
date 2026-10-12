
from pathlib import Path
import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parent

data_path = (
    ROOT / "data" /
    "PS_20174392719_1491204439457_log.csv"
)
model_path = ROOT / "models" / "random_forest.joblib"

bundle = joblib.load(model_path)
model = bundle["pipeline"]
threshold = bundle["threshold"]
features = bundle["features"]

print(f"Model threshold: {threshold:.6f}")

# Load the dataset and select actual fraud cases
df = pd.read_csv(data_path)
max_step = df["step"].max()
test_start = max_step * 0.85

fraud = df[
    (df["isFraud"] == 1) &
    (df["step"] > test_start)
].copy()

if fraud.empty:
    raise ValueError("No fraud cases found in the test period.")

# Score actual fraud cases using the saved model
fraud["fraud_probability"] = model.predict_proba(
    fraud[features]
)[:, 1]

# Keep transactions the model actually flags
flagged = fraud[
    fraud["fraud_probability"] >= threshold
].sort_values("fraud_probability", ascending=False)

if flagged.empty:
    print("No test-set fraud cases exceed the threshold.")
    print("Highest-scoring actual fraud case:")
    example = fraud.sort_values(
        "fraud_probability", ascending=False
    ).iloc[0]
else:
    print(f"Flagged fraud examples found: {len(flagged)}")
    example = flagged.iloc[0]

print("\nEnter these values in the Flask app:")
for col in features:
    print(f"{col}: {example[col]}")

print(f"\nActual fraud label: {int(example['isFraud'])}")
print(
    f"Model fraud probability: "
    f"{example['fraud_probability']:.2%}"
)
print(f"Threshold: {threshold:.2%}")
print(
    "Expected decision:",
    "POTENTIAL FRAUD"
    if example["fraud_probability"] >= threshold
    else "NOT FLAGGED"
)
