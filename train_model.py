# train_model.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import joblib

# Load dataset
df = pd.read_csv("final_airline_times_HHMM.csv")
df.dropna(inplace=True)

# Set target and features
target_column = 'Delayed'
y = df[target_column]
X = df.drop(columns=[target_column])

# Encode categorical features
for col in X.columns:
    if X[col].dtype == 'object':
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))

# Normalize features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Split and train
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model and feature list
joblib.dump(model, "rf_model.pkl")
joblib.dump(X.columns.tolist(), "rf_features.pkl")

print("✅ Model and feature list saved as .pkl files")
