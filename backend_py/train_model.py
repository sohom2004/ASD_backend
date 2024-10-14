import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib

# Sample Data - you would replace this with real gameplay data
# Features: reaction_time_npc, reaction_time_duck, accuracy, impulsivity, task_switch_freq, social_vs_hunting
data = {
    'reaction_time_npc': [3000, 2500, 2800, 3200, 2700],
    'reaction_time_duck': [1200, 1500, 1400, 1300, 1600],
    'accuracy': [85, 92, 88, 81, 90],
    'impulsivity': [10, 8, 9, 11, 7],
    'task_switch_freq': [5, 6, 4, 7, 5],
    'social_vs_hunting': [0.5, 0.6, 0.4, 0.7, 0.55],
    'label': [0, 1, 0, 1, 0]  # Example labels: 1 = ASD, 0 = No ASD
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Features and labels
X = df[['reaction_time_npc', 'reaction_time_duck', 'accuracy', 'impulsivity', 'task_switch_freq', 'social_vs_hunting']]
y = df['label']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling (standardization)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train a Logistic Regression model
model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)

# Model evaluation
y_pred = model.predict(X_test)
print("Accuracy Score:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Save the model and the scaler for later use in the Flask app
joblib.dump(model, 'asddetection_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
