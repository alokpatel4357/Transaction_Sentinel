import matplotlib.pyplot as plt
from xgboost import plot_importance

# Load your trained model (if not already loaded)
import joblib
model = joblib.load('xgb_fraud_model.pkl')

# Plot feature importance
plt.figure(figsize=(10,8))
plot_importance(model, max_num_features=15, importance_type='gain')
plt.title('Top 15 Feature Importances (by Gain)')
plt.show()

import argparse

parser = argparse.ArgumentParser(description='Predict fraud on new transaction data.')
parser.add_argument('--input', type=str, default='new_data.csv', help='Input CSV file with new data')
parser.add_argument('--output', type=str, default='new_data_with_predictions.csv', help='Output CSV file with predictions')
args = parser.parse_args()

# Load data
new_data = pd.read_csv(args.input)

# (rest of your prediction code)

# Save results
new_data.to_csv(args.output, index=False)
print(f"Predictions saved to {args.output}")