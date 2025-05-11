import streamlit as st
import pandas as pd
import joblib

# Load trained model and artifacts
model = joblib.load('flight_delay_model.pkl')
target_le = joblib.load('flight_status_encoder.pkl')
feature_encoders = joblib.load('feature_encoders.pkl')
trained_features_cols = joblib.load('feature_columns.pkl')

# Function to convert HH:MM to minutes
def time_to_minutes(t):
    try:
        h, m = map(int, str(t).strip().split(':'))
        return h * 60 + m
    except:
        st.warning(f"Invalid time format: {t}. Use HH:MM.")
        return 0

# Streamlit Title
st.title("✈️ Flight Delay Predictor")

# Input form
with st.form("flight_form"):
    carrier = st.text_input("Carrier Code (e.g., AA)")
    carrier_name = st.text_input("Carrier Name (e.g., American Airlines)")
    airport = st.text_input("Airport Code (e.g., JFK)")
    airport_name = st.text_input("Airport Name (e.g., John F Kennedy International)")
    scheduled_departure_time = st.text_input("Scheduled Departure Time (HH:MM)")
    scheduled_arrival_time = st.text_input("Scheduled Arrival Time (HH:MM)")
    
    submit = st.form_submit_button("Predict Delay Status")

if submit:
    # Convert times to minutes
    dep_time = time_to_minutes(scheduled_departure_time)
    arr_time = time_to_minutes(scheduled_arrival_time)

    # Construct input DataFrame
    input_data = pd.DataFrame([{
        'carrier': carrier,
        'carrier_name': carrier_name,
        'airport': airport,
        'airport_name': airport_name,
        'scheduled_departure_time': dep_time,
        'scheduled_arrival_time': arr_time
    }])

    # Encode categorical features
    for col in input_data.columns:
        if input_data[col].dtype == 'object' and col in feature_encoders:
            encoder = feature_encoders[col]
            val = input_data.at[0, col]
            if val in encoder.classes_:
                input_data[col] = encoder.transform([val])
            else:
                st.warning(f"Unknown category '{val}' in '{col}'. Assigned 0.")
                input_data[col] = 0

    # Add missing columns that the model was trained on
    for col in trained_features_cols:
        if col not in input_data.columns:
            input_data[col] = 0

    # Reorder to match training feature order
    input_data = input_data[trained_features_cols]

    # Prediction with error handling
    try:
        prediction = model.predict(input_data)
        result = target_le.inverse_transform(prediction)[0]
        st.success(f"🛬 Predicted Flight Status: **{result}**")
    except ValueError as ve:
        st.error("❌ Prediction failed due to a shape mismatch or other input error.")
        st.exception(ve)
