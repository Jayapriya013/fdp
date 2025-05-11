import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load dataset
df = pd.read_csv("final_airline_times_HHMM.csv")

# Convert time columns from HH:MM to integer (e.g., "10:30" → 1030)
def convert_time_to_int(time_str):
    if isinstance(time_str, str) and ":" in time_str:
        h, m = time_str.split(":")
        return int(h) * 100 + int(m)
    return 0

df["scheduled_departure_time"] = df["scheduled_departure_time"].apply(convert_time_to_int)
df["scheduled_arrival_time"] = df["scheduled_arrival_time"].apply(convert_time_to_int)

# Select only the 14 required features
features = [
    "year", "month", "carrier", "carrier_name", "airport", "airport_name",
    "arr_flights", "carrier_ct", "weather_ct", "nas_ct",
    "carrier_delay", "weather_delay", "scheduled_departure_time", "scheduled_arrival_time"
]

X = df[features]
y = df["flight_status"]

# Encode categorical variables
X_encoded = pd.get_dummies(X)

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)  # 'on-time' → 0, 'delayed' → 1

# Save the label encoder
joblib.dump(label_encoder, "models/flight_status_encoder.pkl")

# Save the feature columns for future input alignment
joblib.dump(X_encoded.columns.tolist(), "models/rf_model_columns.pkl")

# Train the model
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y_encoded, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the model
joblib.dump(model, "models/rf_model.pkl")

print("✅ Model retrained and saved successfully.")
