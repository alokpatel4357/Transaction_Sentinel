import joblib
import pandas as pd

# Load the saved model
model = joblib.load('xgb_fraud_model.pkl')

# Load your new data
new_data = pd.read_csv('new_data.csv')

# List of features your model expects
feature_columns = ['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
                   'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20',
                   'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'scaled_amount', 'scaled_time']

# Select features from new data
X_new = new_data[feature_columns]

# Predict fraud probabilities
y_proba_new = model.predict_proba(X_new)[:, 1]

# Use threshold 0.579 to decide fraud or not
threshold = 0.579
y_pred_new = (y_proba_new >= threshold).astype(int)

# Add predictions to your data
new_data['fraud_prediction'] = y_pred_new

# Save the results to a new file
new_data.to_csv('new_data_with_predictions.csv', index=False)

print("Predictions saved to new_data_with_predictions.csv")