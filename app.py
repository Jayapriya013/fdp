import streamlit as st
import pandas as pd
import joblib

# Load model and encoders
model = joblib.load('flight_delay_model.pkl')
target_le = joblib.load('flight_status_encoder.pkl')
feature_encoders = joblib.load('feature_encoders.pkl')
trained_features_cols = joblib.load('feature_columns.pkl')

# Load dataset to fetch valid dropdown options
dataset = pd.read_csv('final_airline_times_HHMM.csv')

# Unique values for dropdowns
carrier_options = sorted(dataset['carrier'].dropna().unique())
carrier_name_options = sorted(dataset['carrier_name'].dropna().unique())
airport_options = sorted(dataset['airport'].dropna().unique())
airport_name_options = sorted(dataset['airport_name'].dropna().unique())

# Helper to convert HH:MM to minutes
def time_to_minutes(t):
    try:
        h, m = map(int, str(t).split(':'))
        return h * 60 + m
    except:
        st.warning(f"Invalid time format: {t}. Use HH:MM.")
        return 0

# App UI
st.title("✈️ Flight Delay Predictor")

with st.form("flight_form"):
    carrier = st.selectbox("Carrier Code", carrier_options)
    carrier_name = st.selectbox("Carrier Name", carrier_name_options)
    airport = st.selectbox("Airport Code", airport_options)
    airport_name = st.selectbox("Airport Name", airport_name_options)
    scheduled_departure_time = st.text_input("Scheduled Departure Time (HH:MM)", "10:00")
    scheduled_arrival_time = st.text_input("Scheduled Arrival Time (HH:MM)", "12:00")
    
    submit = st.form_submit_button("Predict Delay Status")

if submit:
    input_data = {
        'carrier': [carrier],
        'carrier_name': [carrier_name],
        'airport': [airport],
        'airport_name': [airport_name],
        'scheduled_departure_time': [time_to_minutes(scheduled_departure_time)],
        'scheduled_arrival_time': [time_to_minutes(scheduled_arrival_time)],
    }
    
    df_input = pd.DataFrame(input_data)

    # Encode categorical features
    for col in df_input.select_dtypes(include='object').columns:
        if col in feature_encoders:
            encoder = feature_encoders[col]
            df_input[col] = encoder.transform(df_input[col])
        else:
            df_input[col] = 0

    # Ensure all required columns are present
    for col in trained_features_cols:
        if col not in df_input.columns:
            df_input[col] = 0

    # Reorder columns
    df_input = df_input[trained_features_cols]

    try:
        prediction = model.predict(df_input)
        result = target_le.inverse_transform(prediction)[0]
        st.success(f"🛬 Predicted Flight Status: **{result}**")
    except Exception as e:
        st.error(f"❌ Prediction failed: {str(e)}")
