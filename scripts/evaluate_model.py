import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Load the saved model
model = joblib.load('models/xgb_fraud_model.pkl')

# Load test data (replace with your actual test data file)
test_data = pd.read_csv('data/test_data.csv')

# Define feature columns (same as training)
feature_columns = ['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
                   'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20',
                   'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'scaled_amount', 'scaled_time']

# Extract features and true labels
X_test = test_data[feature_columns]
y_test = test_data['Class']  # assuming 'Class' column has true labels

# Predict probabilities for the positive class (fraud)
y_proba_test = model.predict_proba(X_test)[:, 1]

# Set threshold (same as used in prediction)
threshold = 0.579

# Convert probabilities to binary predictions
y_pred_test = (y_proba_test >= threshold).astype(int)

# Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_pred_test)
precision = precision_score(y_test, y_pred_test)
recall = recall_score(y_test, y_pred_test)
f1 = f1_score(y_test, y_pred_test)
roc_auc = roc_auc_score(y_test, y_proba_test)

# Print results
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")
print(f"ROC AUC:   {roc_auc:.4f}")
