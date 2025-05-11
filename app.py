import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load trained model and expected feature order
model = joblib.load("rf_model.pkl")
expected_features = joblib.load("rf_features.pkl")  # Must match what you saved during training

st.title("Flight Delay Prediction")

# UI inputs
feature_input = {
    "CRS Elapsed Time (mins)": st.number_input("CRS Elapsed Time (mins)", 0),
    "Distance": st.number_input("Distance (miles)", 0),
    "NAS CT": st.number_input("NAS CT", 0),
    "Carrier Delay (mins)": st.number_input("Carrier Delay (mins)", 0),
    "Weather Delay (mins)": st.number_input("Weather Delay (mins)", 0),
    "Scheduled Departure Time (HH:MM)": st.text_input("Scheduled Departure Time (HH:MM)", "10:00"),
    "Scheduled Arrival Time (HH:MM)": st.text_input("Scheduled Arrival Time (HH:MM)", "12:00"),
    # Add other features here to match training input
}

# Optional: Convert HH:MM time to minutes (if model used that)
def time_to_minutes(tstr):
    h, m = map(int, tstr.split(":"))
    return h * 60 + m

# Include any time conversion if your model used it
feature_input["Scheduled Departure Time (mins)"] = time_to_minutes(feature_input.pop("Scheduled Departure Time (HH:MM)"))
feature_input["Scheduled Arrival Time (mins)"] = time_to_minutes(feature_input.pop("Scheduled Arrival Time (HH:MM)"))

# Convert to DataFrame and align with expected features
input_df = pd.DataFrame([feature_input])
input_df = input_df.reindex(columns=expected_features, fill_value=0)

if st.button("Predict"):
    try:
        prediction = model.predict(input_df)[0]
        result = "Delayed" if prediction == 1 else "On-Time"
        st.success(f"Prediction: ✈️ {result}")
    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")
