# Fraudulent Transaction Detection System

A machine learning project that identifies potentially fraudulent financial transactions using Logistic Regression and Random Forest. The project includes data preprocessing, class-imbalance handling, model evaluation, and a Flask web interface for individual transaction scoring.

## Project Overview

Financial fraud detection is challenging because fraudulent transactions represent a very small fraction of all transactions. A model can achieve high accuracy simply by predicting every transaction as legitimate.

This project uses the PaySim synthetic financial transaction dataset to investigate fraud detection under severe class imbalance.

**Key highlights**
- Analysed a dataset containing approximately 6.36 million transactions.
- Implemented chronological training, validation, and test splits.
- Applied random undersampling to the training set at a 10:1 legitimate-to-fraud ratio.
- Trained and compared Logistic Regression and Random Forest.
- Evaluated precision, recall, F1-score, ROC-AUC, average precision, and confusion matrix counts.
- Built a Flask web application for individual transaction scoring.

## Dataset

**Source:** [PaySim — Synthetic Financial Datasets for Fraud Detection](https://www.kaggle.com/datasets/ealaxi/paysim1)

The dataset contains 6,362,620 simulated mobile-money transactions, including 8,213 labelled fraudulent transactions.

The original fraud rate is approximately 0.1291%.

Download the dataset from Kaggle and place the extracted CSV in:

`data/PS_20174392719_1491204439457_log.csv`

The dataset is not included in this repository.

### Features used

| Feature | Description |
|---|---|
| `step` | Transaction time step |
| `type` | Transaction type |
| `amount` | Transaction amount |
| `oldbalanceOrg` | Sender's balance before the transaction |
| `oldbalanceDest` | Recipient's balance before the transaction |

The target variable is `isFraud`, where 1 indicates a labelled fraudulent transaction and 0 indicates a legitimate transaction.

Sender and recipient identifiers, post-transaction balances, and the pre-existing `isFlaggedFraud` field are excluded from the model features.

## Methodology

### 1. Data preparation
- Loaded the transaction dataset and validated the required columns.
- Removed rows with missing values in required fields.
- Divided the data chronologically into training, validation, and test sets.

### 2. Class-imbalance handling
Random undersampling was applied to the training set only. All available fraud cases in the training period were retained, while legitimate transactions were sampled at up to ten times the number of fraud cases.

The validation and test sets retained their natural class distributions.

### 3. Model development
Two classifiers were trained using a scikit-learn pipeline:

- **Logistic Regression:** baseline classification model.
- **Random Forest:** ensemble classifier used to capture nonlinear relationships.

Numerical features were standardised, and transaction type was one-hot encoded.

### 4. Threshold selection
The classification threshold was selected using validation data to target at least 95% recall, choosing the highest eligible threshold. Final metrics were then calculated on the held-out test set.

### 5. Evaluation
Models were compared using:
- Precision
- Recall
- F1-score
- ROC-AUC
- Average precision
- Confusion matrix counts

## Results

The following results were obtained in the current experiment.

| Metric | Logistic Regression | Random Forest |
|---|---:|---:|
| Precision | 3.88% | 38.29% |
| Recall | 95.85% | 95.45% |
| F1-score | 7.46% | 54.65% |
| ROC-AUC | 94.45% | 99.67% |
| Average precision | 58.00% | 88.45% |

Random Forest was selected as the preferred model because it achieved substantially higher precision and F1-score while maintaining similar recall.

### Random Forest confusion matrix

| Outcome | Transactions |
|---|---:|
| True negatives | 86,288 |
| False positives | 1,926 |
| False negatives | 57 |
| True positives | 1,195 |

The model detected 1,195 of the 1,252 fraudulent transactions in the test set. It also incorrectly flagged 1,926 legitimate transactions.

**Interpretation:** High recall helps reduce missed fraud, but precision shows that many alerts still require further review. The appropriate operating threshold depends on the relative costs of missed fraud and false alarms.

### Evaluation visualisations

The evaluation script generates the following charts in the `reports/` directory:

- `confusion_matrix.png`
- `precision_recall_curve.png`

The model comparison table is saved as `model_comparison.csv`, and the detailed metrics are stored in `metrics.json`.

## Web Application

The Flask application allows users to enter:
- Transaction time step
- Transaction type
- Transaction amount
- Sender's previous balance
- Recipient's previous balance

It returns the model's estimated fraud probability and whether the transaction crosses the configured decision threshold.

The application is an educational prototype for individual transaction scoring. It is not connected to a live payment network and has not been validated for production financial decisions.

## Installation and Usage

### 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd fraud-detection-system
```

### 2. Create a virtual environment

On Windows:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
python -m pip install -r requirements.txt
```

If the global `python` command is unavailable, use:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### 4. Download the dataset

Download PaySim from Kaggle and place the extracted CSV in the `data/` directory using the filename specified above.

### 5. Train the models

```powershell
python train.py
```

This saves the trained models in `models/` and evaluation results in `reports/`.

### 6. Generate evaluation charts

```powershell
python evaluate_model.py
```

### 7. Launch the Flask application

```powershell
python app.py
```

Open [http://127.0.0.1:5000](http://127.0.0.1:5000) in your browser.

## Project Structure

```text
fraud-detection-system/
├── data/
│   └── PS_20174392719_1491204439457_log.csv
├── models/
│   ├── logistic_regression.joblib
│   └── random_forest.joblib
├── reports/
│   ├── confusion_matrix.png
│   ├── precision_recall_curve.png
│   ├── model_comparison.csv
│   └── metrics.json
├── train.py
├── evaluate_model.py
├── find_fraud_example.py
├── app.py
├── requirements.txt
├── .gitignore
└── README.md
```

Generated model files and reports appear after the relevant scripts are run.

## Limitations and Future Improvements

- PaySim contains simulated mobile-money transactions, not real credit-card transactions.
- Fraud prevalence changes substantially across the chronological data splits. This indicates a potential distribution shift that needs further investigation.
- Undersampling discards many legitimate training transactions and may affect generalisation.
- The selected threshold targets recall on validation data; performance can change on later transactions.
- The model uses five features and does not include a live payment-processing integration.
- Cost savings have not been verified in a real financial institution.

Future improvements include investigating temporal drift, testing additional features, comparing resampling and class-weighting approaches, calibrating predicted probabilities, and conducting cost-sensitive threshold optimisation.

## Technologies

Python · pandas · NumPy · scikit-learn · Matplotlib · Flask · joblib · Git

## Disclaimer

This project is for educational and portfolio purposes. Predictions are model estimates and should not be interpreted as proof of fraudulent activity.