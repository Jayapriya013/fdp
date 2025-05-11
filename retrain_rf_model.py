import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load data
df = pd.read_csv("final_airline_times_HHMM.csv")

# Define features and target
features = [
    "year", "month", "carrier", "carrier_name", "airport", "airport_name",
    "arr_flights", "carrier_ct", "weather_ct", "nas_ct",
    "carrier_delay", "weather_delay", "scheduled_departure_time", "scheduled_arrival_time"
]
target = "status"  # Make sure this column exists in your CSV

X = df[features]
y = df[target]

# Encode categorical features
X_encoded = pd.get_dummies(X)
joblib.dump(X_encoded.columns.tolist(), "feature_columns.pkl")

# Train model
model = RandomForestClassifier()
model.fit(X_encoded, y)

# Save model
joblib.dump(model, "flight_delay_model.pkl")
print("✅ Model trained and saved with 14 features.")
