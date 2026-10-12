
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, request, render_template_string

app = Flask(__name__)

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "random_forest.joblib"

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        "Random Forest model not found. "
        "Run train.py first."
    )

# Load the saved pipeline and its validation-selected threshold.
bundle = joblib.load(MODEL_PATH)

model = bundle["pipeline"]
threshold = float(bundle["threshold"])
features = bundle["features"]

TRANSACTION_TYPES = [
    "CASH_IN",
    "CASH_OUT",
    "DEBIT",
    "PAYMENT",
    "TRANSFER"
]

HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Fraud Detection System</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            background: #f2f5f9;
            margin: 0;
            padding: 30px 16px;
            color: #1f2937;
        }
        .container {
            max-width: 650px;
            margin: auto;
            background: white;
            padding: 28px;
            border-radius: 14px;
            box-shadow: 0 5px 22px #00000012;
        }
        h1 { margin-top: 0; color: #17365d; }
        .subtitle { color: #64748b; line-height: 1.5; }
        label {
            display: block;
            font-weight: bold;
            margin: 17px 0 7px;
        }
        input, select {
            width: 100%;
            box-sizing: border-box;
            padding: 11px;
            border: 1px solid #cbd5e1;
            border-radius: 7px;
            font-size: 15px;
        }
        button {
            width: 100%;
            padding: 13px;
            margin-top: 24px;
            border: 0;
            border-radius: 8px;
            background: #17365d;
            color: white;
            font-size: 16px;
            cursor: pointer;
        }
        button:hover { background: #254f80; }
        .result {
            padding: 17px;
            margin-top: 24px;
            border-radius: 9px;
            background: #f1f5f9;
            line-height: 1.7;
            overflow-wrap: anywhere;
        }
        .fraud { border-left: 5px solid #dc2626; }
        .safe { border-left: 5px solid #16a34a; }
        .error { border-left: 5px solid #d97706; }
        .small { font-size: 12px; color: #64748b; }
    </style>
</head>
<body>
<div class="container">
    <h1>Fraud Detection System</h1>
    <p class="subtitle">
        Enter transaction details to obtain a prediction
        from the trained Random Forest model.
    </p>

    <form method="POST">
        <label for="step">Transaction step</label>
        <input id="step" name="step" type="number"
               min="1" max="744" step="1"
               value="{{ values.get('step', '100') }}" required>

        <label for="type">Transaction type</label>
        <select id="type" name="type" required>
            {% for t in transaction_types %}
            <option value="{{ t }}"
                {% if values.get('type', 'TRANSFER') == t %}
                selected
                {% endif %}>
                {{ t }}
            </option>
            {% endfor %}
        </select>

        <label for="amount">Transaction amount</label>
        <input id="amount" name="amount" type="number"
               min="0" step="any"
               value="{{ values.get('amount', '1000') }}" required>

        <label for="oldbalanceOrg">Sender's previous balance</label>
        <input id="oldbalanceOrg" name="oldbalanceOrg"
               type="number" min="0" step="any"
               value="{{ values.get('oldbalanceOrg', '5000') }}"
               required>

        <label for="oldbalanceDest">Recipient's previous balance</label>
        <input id="oldbalanceDest" name="oldbalanceDest"
               type="number" min="0" step="any"
               value="{{ values.get('oldbalanceDest', '2000') }}"
               required>

        <button type="submit">Analyze Transaction</button>
    </form>

    {% if result %}
    <div class="result {{ result.css_class }}">
        <strong>{{ result.title }}</strong><br>
        Fraud probability: {{ result.probability }}%<br>
        Decision threshold: {{ result.threshold }}%<br>
        {{ result.message }}
    </div>
    {% endif %}

    {% if error %}
    <div class="result error">{{ error }}</div>
    {% endif %}

    <p class="small">
        Educational prototype using synthetic PaySim data.
        A prediction is not proof of fraud. This demo is not
        connected to a payment network and should not be used
        to make actual financial decisions.
    </p>
</div>
</body>
</html>
"""


@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    error = None
    values = request.form.to_dict() if request.method == "POST" else {}

    if request.method == "POST":
        try:
            step = int(values["step"])
            transaction_type = values["type"]
            amount = float(values["amount"])
            sender_balance = float(values["oldbalanceOrg"])
            recipient_balance = float(values["oldbalanceDest"])

            if not 1 <= step <= 744:
                raise ValueError("Step must be between 1 and 744.")

            if transaction_type not in TRANSACTION_TYPES:
                raise ValueError("Invalid transaction type.")

            numbers = [amount, sender_balance, recipient_balance]

            if not all(pd.notna(x) and x >= 0 for x in numbers):
                raise ValueError(
                    "Amounts and balances must be finite, "
                    "non-negative numbers."
                )

            if not all(
                abs(x) < float("inf") for x in numbers
            ):
                raise ValueError("Please enter finite numbers.")

            transaction = pd.DataFrame([{
                "step": step,
                "type": transaction_type,
                "amount": amount,
                "oldbalanceOrg": sender_balance,
                "oldbalanceDest": recipient_balance
            }])[features]

            probability = float(
                model.predict_proba(transaction)[0, 1]
            )

            flagged = probability >= threshold

            if flagged:
                title = "Potential fraud detected"
                css_class = "fraud"
                message = (
                    "The transaction meets the model's "
                    "configured alert threshold."
                )
            else:
                title = "Not flagged by the model"
                css_class = "safe"
                message = (
                    "The transaction is below the alert "
                    "threshold; this does not guarantee "
                    "that it is legitimate."
                )

            result = {
                "title": title,
                "css_class": css_class,
                "probability": round(probability * 100, 2),
                "threshold": round(threshold * 100, 2),
                "message": message
            }

        except (ValueError, KeyError, TypeError) as exc:
            error = str(exc) or "Please enter valid transaction details."

    return render_template_string(
        HTML,
        result=result,
        error=error,
        values=values,
        transaction_types=TRANSACTION_TYPES
    )


if __name__ == "__main__":
    app.run(debug=False, host="127.0.0.1", port=5000)
