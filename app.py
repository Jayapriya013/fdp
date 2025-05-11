# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

st.title("✈️ Flight Delay Prediction App")

# Load model & feature list
if not os.path.exists("rf_model.pkl") or not os.path.exists("rf_features.pkl"):
    st.error("Model files not found. Run `train_model.py` first.")
else:
    model = joblib.load("rf_model.pkl")
    expected_features = joblib.load("rf_features.pkl")

    st.header("Enter Flight Details")

    input_dict = {
        "CRS Elapsed Time (mins)": st.number_input("CRS Elapsed Time (mins)", 0),
        "Distance": st.number_input("Distance (miles)", 0),
        "NAS CT": st.number_input("NAS CT", 0),
        "Carrier Delay (mins)": st.number_input("Carrier Delay (mins)", 0),
        "Weather Delay (mins)": st.number_input("Weather Delay (mins)", 0),
        "Scheduled Departure Time (HH:MM)": st.text_input("Departure Time", "10:00"),
        "Scheduled Arrival Time (HH:MM)": st.text_input("Arrival Time", "12:00"),
    }

    # Convert time to minutes
    def time_to_minutes(tstr):
        try:
            h, m = map(int, tstr.split(":"))
            return h * 60 + m
        except:
            return 0

    input_dict["Scheduled Departure Time (mins)"] = time_to_minutes(input_dict.pop("Scheduled Departure Time (HH:MM)"))
    input_dict["Scheduled Arrival Time (mins)"] = time_to_minutes(input_dict.pop("Scheduled Arrival Time (HH:MM)"))

    # Convert to DataFrame
    input_df = pd.DataFrame([input_dict])
    input_df = input_df.reindex(columns=expected_features, fill_value=0)

    if st.button("Predict"):
        prediction = model.predict(input_df)[0]
        result = "Delayed" if prediction == 1 else "On-Time"
        st.success(f"🧾 Prediction: **{result}**")
