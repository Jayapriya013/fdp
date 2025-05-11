# predict_with_dataset.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import joblib

# 1. Load dataset
df = pd.read_csv("final_airline_times_HHMM.csv")
df.dropna(inplace=True)

# 2. Target and features
target_column = 'Delayed'
y = df[target_column]
X = df.drop(columns=[target_column])

# 3. Encode string columns
for col in X.columns:
    if X[col].dtype == 'object':
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))

# 4. Scale numeric features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# 5. Train model
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 6. Predict on all data
predictions = model.predict(X_scaled)
labels = ["Delayed" if p == 1 else "On-Time" for p in predictions]

# 7. Save predictions to CSV
df["Predicted_Status"] = labels
df.to_csv("predicted_flight_delays.csv", index=False)

# 8. Save model and feature list
joblib.dump(model, "rf_model.pkl")
joblib.dump(X.columns.tolist(), "rf_features.pkl")
print("✅ All files saved.")
