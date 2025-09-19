import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Step 1: Load the dataset
df = pd.read_csv('creditcard.csv')

# Step 2: Scale 'Amount' and 'Time'
scaler = StandardScaler()
df['scaled_amount'] = scaler.fit_transform(df['Amount'].values.reshape(-1, 1))
df['scaled_time'] = scaler.fit_transform(df['Time'].values.reshape(-1, 1))

# Step 3: Drop original 'Amount' and 'Time'
df = df.drop(['Amount', 'Time'], axis=1)

# Step 4: Define features and target
X = df.drop('Class', axis=1)
y = df['Class']

# Step 5: Split into train and test sets with stratification
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# Step 6: Print dataset info
print(f'Training samples: {X_train.shape[0]}')
print(f'Testing samples: {X_test.shape[0]}')
print(f'Fraud cases in training set: {sum(y_train)}')
print(f'Fraud cases in testing set: {sum(y_test)}')
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score

# Initialize Random Forest with balanced class weights
model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)

# Train the model
model.fit(X_train, y_train)

# Predict on test set
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

# Print classification report
print(classification_report(y_test, y_pred))

# Print ROC AUC score
print('ROC AUC score:', roc_auc_score(y_test, y_proba))

from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, roc_auc_score

# Apply SMOTE to balance the training data
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

print(f'After SMOTE, training samples: {X_train_smote.shape[0]}')
print(f'Fraud cases after SMOTE: {sum(y_train_smote)}')

# Initialize XGBoost classifier
xgb_model = XGBClassifier(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    scale_pos_weight=1,  # SMOTE balances classes, so no need to scale pos weight
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42
)

# Train the model on SMOTE data
xgb_model.fit(X_train_smote, y_train_smote)

# Predict on test set
y_pred_xgb = xgb_model.predict(X_test)
y_proba_xgb = xgb_model.predict_proba(X_test)[:, 1]

# Print classification report
print("XGBoost Classification Report:")
print(classification_report(y_test, y_pred_xgb))

# Print ROC AUC score
print('XGBoost ROC AUC score:', roc_auc_score(y_test, y_proba_xgb))

import numpy as np
from sklearn.metrics import precision_recall_curve, classification_report

# Get predicted probabilities for the positive class
y_scores = y_proba_xgb

# Calculate precision-recall curve
precision, recall, thresholds = precision_recall_curve(y_test, y_scores)

# Find threshold with precision >= 0.8 and recall >= 0.7 (example targets)
for p, r, t in zip(precision, recall, thresholds):
    if p >= 0.8 and r >= 0.7:
        print(f"Chosen threshold: {t:.3f}, Precision: {p:.3f}, Recall: {r:.3f}")
        chosen_threshold = t
        break
else:
    # If no threshold meets criteria, pick threshold with best F1 score
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-6)
    best_idx = np.argmax(f1_scores)
    chosen_threshold = thresholds[best_idx]
    print(f"No threshold met criteria, using best F1 threshold: {chosen_threshold:.3f}")

# Apply chosen threshold to predictions
y_pred_threshold = (y_scores >= chosen_threshold).astype(int)

# Print classification report with new threshold
print("Classification report with adjusted threshold:")
print(classification_report(y_test, y_pred_threshold))

from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier
import numpy as np

# Define parameter grid for tuning
param_dist = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7, 10],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'min_child_weight': [1, 3, 5],
    'gamma': [0, 0.1, 0.3],
}

# Initialize XGBoost model
xgb = XGBClassifier(
    use_label_encoder=False,
    eval_metric='logloss',
    scale_pos_weight=1,
    random_state=42
)

# Setup RandomizedSearchCV
random_search = RandomizedSearchCV(
    estimator=xgb,
    param_distributions=param_dist,
    n_iter=20,  # number of parameter settings sampled
    scoring='roc_auc',
    cv=3,
    verbose=2,
    random_state=42,
    n_jobs=-1
)

# Fit on SMOTE training data
random_search.fit(X_train_smote, y_train_smote)

print("Best parameters found: ", random_search.best_params_)
print("Best ROC AUC score: ", random_search.best_score_)

# Use best estimator to predict on test set
best_model = random_search.best_estimator_
y_proba_best = best_model.predict_proba(X_test)[:, 1]

# Adjust threshold as before (optional)
from sklearn.metrics import precision_recall_curve, classification_report

precision, recall, thresholds = precision_recall_curve(y_test, y_proba_best)

# Example: choose threshold with precision >= 0.8 and recall >= 0.7
for p, r, t in zip(precision, recall, thresholds):
    if p >= 0.8 and r >= 0.7:
        chosen_threshold = t
        print(f"Chosen threshold: {t:.3f}, Precision: {p:.3f}, Recall: {r:.3f}")
        break
else:
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-6)
    best_idx = np.argmax(f1_scores)
    chosen_threshold = thresholds[best_idx]
    print(f"No threshold met criteria, using best F1 threshold: {chosen_threshold:.3f}")

y_pred_best = (y_proba_best >= chosen_threshold).astype(int)

print("Classification report with tuned model and adjusted threshold:")
print(classification_report(y_test, y_pred_best))

import joblib

joblib.dump(best_model, 'xgb_fraud_model.pkl')
print("Model saved as xgb_fraud_model.pkl")