import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load your dataset
df = pd.read_csv("final_airline_times_HHMM.csv")

# Convert HH:MM to integer format
def convert_time_to_int(time_str):
    try:
        h, m = map(int, str(time_str).split(':'))
        return h * 100 + m
    except:
        return 0

df["scheduled_departure_time"] = df["scheduled_departure_time"].apply(convert_time_to_int)
df["scheduled_arrival_time"] = df["scheduled_arrival_time"].apply(convert_time_to_int)

# Define 14 features
features = [
    "year", "month", "carrier", "carrier_name", "airport", "airport_name",
    "arr_flights", "carrier_ct", "weather_ct", "nas_ct",
    "carrier_delay", "weather_delay", "scheduled_departure_time", "scheduled_arrival_time"
]

X = df[features]
y = df["flight_status"]

# Encode target label
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
joblib.dump(label_encoder, "flight_status_encoder.pkl")

# One-hot encode categorical features
X_encoded = pd.get_dummies(X)

# Save column list for prediction use
joblib.dump(X_encoded.columns.tolist(), "rf_model_columns.pkl")

# Train model
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y_encoded, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the model
joblib.dump(model, "rf_model.pkl")
print("✅ Model trained and saved successfully.")
