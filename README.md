# Transaction Sentinel – Fraud Detection System

### Description

A real-time, AI-powered system to combat financial fraud. The model analyzes transaction patterns, user behavior (time of day, location, purchase amount), and device data to instantly flag suspicious activity. The solution is trained on a dataset of transactions to learn what constitutes normal behavior for a user and can detect anomalies with high accuracy, minimizing false positives and protecting both customers and businesses from financial loss.

---

### Key Requirements

- A real-time data processing pipeline capable of handling high volumes of transactions
- An anomaly detection model trained on user-specific and general transaction patterns
- Secure integration with financial systems and payment gateways
- A dashboard for fraud analysts to review and act on flagged transactions
- Compliance with financial data security standards (e.g., PCI DSS)

### Outcomes

- Quantifiable reduction in fraudulent transaction losses
- Low false-positive rate to ensure legitimate transactions are not blocked unnecessarily
- Improved trust and security for customers

### Conditions

- Access to large volumes of anonymized transaction data is essential for training the model
- The system must operate with very low latency to approve or flag transactions in real-time
- The model must be adaptable to changing fraud patterns

---

## Datasets

- **Kaggle Credit Card Fraud Detection:**  
  [Link on Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **IEEE-CIS Fraud Detection:**  
  [Link on Kaggle](https://www.kaggle.com/c/ieee-fraud-detection)

---

## Setup

Install dependencies:

pip install -r requirements.txt

## Usage

Predict on new data:

---

python3 scripts/predict_fraud.py --input data/new_data.csv --output data/predictions.csv

Evaluate on test data:

---

python3 scripts/evaluate_model.py


## Data

- `data/test_data.csv`: Sample labeled test data
- `data/new_data.csv`: New transactions for prediction
- **`creditcard.csv`**: The full dataset is not included due to GitHub file size limits.

### How to Add `creditcard.csv`

1. Download it from [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
2. Place `creditcard.csv` in the root directory of this project (`Transaction_Sentinel/creditcard.csv`)

> For testing, sample CSVs are already provided in the `/data` folder.

---

## Model

- `models/xgb_fraud_model.pkl`: Trained XGBoost model

## Scripts

- `scripts/predict_fraud.py`: Prediction script
- `scripts/evaluate_model.py`: Evaluation script
- `scripts/feature_importance.py`: Visualizes feature importances (optional)

## Notebooks

- `notebooks/Transaction_Sentinel_Prediction.ipynb`: Jupyter notebook for step-by-step demonstration (optional)

## License

Distributed for educational and research purposes.
