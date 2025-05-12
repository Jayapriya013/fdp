import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Load model and feature columns
@st.cache_data
def load_model():
    with open('model/flight_delay_model.pkl', 'rb') as f:
        model, feature_columns = pickle.load(f)
    return model, feature_columns

model, feature_columns = load_model()

st.title("✈️ Flight Delay Prediction App")
st.write("Enter flight details to predict if the flight will be delayed.")

# Input fields
airline = st.selectbox("Airline", ["Delta", "United", "Southwest", "JetBlue"])
day_of_week = st.selectbox("Day of Week", list(range(1, 8)))
departure_hour = st.slider("Scheduled Departure Hour", 0, 23, 12)
origin = st.selectbox("Origin Airport", ["JFK", "LAX", "ATL", "ORD"])
destination = st.selectbox("Destination Airport", ["SFO", "SEA", "MIA", "DFW"])
distance = st.number_input("Flight Distance (miles)", min_value=50, max_value=3000, value=500)
wind_speed = st.slider("Wind Speed (mph)", 0, 100, 10)
precipitation = st.slider("Precipitation (inches)", 0.0, 2.0, 0.0)

# Create DataFrame for input
input_dict = {
    'Airline': airline,
    'DayOfWeek': day_of_week,
    'DepHour': departure_hour,
    'Origin': origin,
    'Dest': destination,
    'Distance': distance,
    'WindSpeed': wind_speed,
    'Precipitation': precipitation
}
input_df = pd.DataFrame([input_dict])

# One-hot encode and align columns
input_encoded = pd.get_dummies(input_df)
missing_cols = set(feature_columns) - set(input_encoded.columns)
for col in missing_cols:
    input_encoded[col] = 0
input_encoded = input_encoded[feature_columns]  # Reorder

# Predict
if st.button("Predict Delay"):
    prediction = model.predict(input_encoded)[0]
    proba = model.predict_proba(input_encoded)[0][1]

    if prediction == 1:
        st.error(f"⚠️ Flight is likely to be **delayed** (probability: {proba:.2f})")
    else:
        st.success(f"✅ Flight is likely to be **on time** (probability: {1 - proba:.2f})")
