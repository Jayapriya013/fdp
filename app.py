# app.py
import streamlit as st
import pandas as pd
import joblib
import os

st.title("✈️ Flight Delay Predictor")

# Load model
if not os.path.exists("rf_model.pkl") or not os.path.exists("rf_features.pkl"):
    st.error("❌ Model files not found. Run `predict_with_dataset.py` first.")
else:
    model = joblib.load("rf_model.pkl")
    expected_features = joblib.load("rf_features.pkl")

    st.subheader("Enter Flight Info")

    input_dict = {
        "airport_name(mins)": st.number_input("airport_name", 0),
        "nas_ct": st.number_input("ct", 0),
        "carrier_name": st.number_input("carrier_name", 0),
        "weather_delay (mins)": st.number_input("weather_delay", 0),
        "scheduled_departure_time (HH:MM)": st.text_input("scheduled_departure_time", "10:00"),
        "scheduled_arrival_time (HH:MM)": st.text_input("scheduled_arrival_time", "12:00"),
    }

    # Convert time fields
    def time_to_minutes(tstr):
        try:
            h, m = map(int, tstr.split(":"))
            return h * 60 + m
        except:
            return 0

    input_dict["scheduled_dparture_time (mins)"] = time_to_minutes(input_dict.pop("scheduled_departure_time (HH:MM)"))
    input_dict["scheduled_arrival_time (mins)"] = time_to_minutes(input_dict.pop("scheduled_arrival_time (HH:MM)"))

    # Create input DataFrame
    input_df = pd.DataFrame([input_dict])
    input_df = input_df.reindex(columns=expected_features, fill_value=0)

    if st.button("Predict"):
        pred = model.predict(input_df)[0]
        result = "✈️ Delayed" if pred == 1 else "✅ On-Time"
        st.success(f"Prediction: **{result}**")
