import streamlit as st
import pandas as pd
import joblib

# Load model and encoders
model = joblib.load('flight_delay_model.pkl')
target_le = joblib.load('flight_status_encoder.pkl')
feature_encoders = joblib.load('feature_encoders.pkl')
trained_features_cols = joblib.load('feature_columns.pkl')

# Function to convert HH:MM to minutes
def time_to_minutes(t):
    try:
        h, m = map(int, str(t).split(':'))
        return h * 60 + m
    except:
        st.warning(f"Invalid time format: {t}. Use HH:MM.")
        return 0

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
    # Build input data
    data = {
        'carrier': [carrier],
        'carrier_name': [carrier_name],
        'airport': [airport],
        'airport_name': [airport_name],
        'scheduled_departure_time': [time_to_minutes(scheduled_departure_time)],
        'scheduled_arrival_time': [time_to_minutes(scheduled_arrival_time)]
    }

    df_new = pd.DataFrame(data)

    # Encode categorical features
    for col in df_new.select_dtypes(include='object').columns:
        if col in feature_encoders:
            encoder = feature_encoders[col]
            def encode_value(val):
                val_str = str(val)
                if val_str in encoder.classes_:
                    return encoder.transform([val_str])[0]
                else:
                    st.warning(f"Unknown category '{val_str}' in column '{col}', assigning 0")
                    return 0
            df_new[col] = df_new[col].apply(encode_value)

    # Add any missing columns used in training
    for col in trained_features_cols:
        if col not in df_new.columns:
            df_new[col] = 0  # Fill missing with default 0

    # Reorder columns
    df_new = df_new[trained_features_cols]

    # Make prediction
    prediction = model.predict(df_new)
    result = target_le.inverse_transform(prediction)[0]

    st.success(f"🛬 Predicted Flight Status: **{result}**")
