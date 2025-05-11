git add predict_delay.py
git commit -m "Add prediction script using all 14 features"
git push origin main
import pandas as pd
import joblib

def predict_flight_delay(input_data):
    # Load model and columns
    model = joblib.load("models/rf_model.pkl")
    model_columns = joblib.load("models/rf_model_columns.pkl")

    # Convert to DataFrame
    input_df = pd.DataFrame(input_data)

    # Encode categorical features
    input_encoded = pd.get_dummies(input_df)

    # Align with model columns
    input_encoded = input_encoded.reindex(columns=model_columns, fill_value=0)

    # Predict
    prediction = model.predict(input_encoded)
    return prediction[0]

if __name__ == "__main__":
    # Example test
    test_input = {
        "year": [2025],
        "month": [5],
        "carrier": ["9E"],
        "carrier_name": ["Endeavor Air Inc."],
        "airport": ["ABR"],
        "airport_name": ["Aberdeen Regional"],
        "arr_flights": [12],
        "carrier_ct": [5],
        "weather_ct": [0],
        "nas_ct": [1],
        "carrier_delay": [0],
        "weather_delay": [0],
        "scheduled_departure_time": [1000],
        "scheduled_arrival_time": [1200],
    }

    result = predict_flight_delay(test_input)
    print("Prediction:", result)
