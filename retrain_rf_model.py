import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
import os

# === Step 1: Load your dataset ===
df = pd.read_csv("data/your_dataset.csv")  # Change path if needed

# === Step 2: Define the 14 input features ===
features = [
    "year", "month", "carrier", "carrier_name", "airport", "airport_name",
    "arr_flights", "carrier_ct", "weather_ct", "nas_ct",
    "carrier_delay", "weather_delay",
    "scheduled_departure_time", "scheduled_arrival_time"
]

# === Step 3: Target column ===
target = "delay_label"  # Change this to match your dataset

# === Step 4: One-hot encode categorical columns ===
X = pd.get_dummies(df[features])
y = df[target]

# === Step 5: Train-test split ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === Step 6: Train the Random Forest model ===
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# === Step 7: Save model and feature columns ===
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/rf_model.pkl")
joblib.dump(X.columns.tolist(), "models/rf_model_columns.pkl")

print("✅ Model trained and saved successfully.")
