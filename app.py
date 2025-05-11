import streamlit as st
import pandas as pd
import joblib

st.title("✈️ Flight Delay Predictor")

# Load encoders and model
model = joblib.load("flight_delay_model.pkl")
model_columns = joblib.load("feature_columns.pkl")
df = pd.read_csv("final_airline_times_HHMM.csv")

with st.form("flight_form"):
    year = st.number_input("Year", value=2025)
    month = st.number_input("Month", min_value=1, max_value=12, value=5)
    carrier = st.selectbox("Carrier", sorted(df["carrier"].dropna().unique()))
    carrier_name = st.selectbox("Carrier Name", sorted(df["carrier_name"].dropna().unique()))
    airport = st.selectbox("Airport Code", sorted(df["airport"].dropna().unique()))
    airport_name = st.selectbox("Airport Name", sorted(df["airport_name"].dropna().unique()))
    arr_flights = st.number_input("Arriving Flights", value=10)
    carrier_ct = st.number_input("Carrier CT", value=0)
    weather_ct = st.number_input("Weather CT", value=0)
    nas_ct = st.number_input("NAS CT", value=0)
    carrier_delay = st.number_input("Carrier Delay (mins)", value=0)
    weather_delay = st.number_input("Weather Delay (mins)", value=0)
    departure_time = st.text_input("Scheduled Departure Time (HH:MM)", "10:00")
    arrival_time = st.text_input("Scheduled Arrival Time (HH:MM)", "12:00")

    submit = st.form_submit_button("Predict")

# Helper: convert HH:MM to minutes
def time_to_minutes(t):
    try:
        h, m = map(int, str(t).split(':'))
        return h * 60 + m
    except:
        return 0

if submit:
    input_data = {
        "year": [year],
        "month": [month],
        "carrier": [carrier],
        "carrier_name": [carrier_name],
        "airport": [airport],
        "airport_name": [airport_name],
        "arr_flights": [arr_flights],
        "carrier_ct": [carrier_ct],
        "weather_ct": [weather_ct],
        "nas_ct": [nas_ct],
        "carrier_delay": [carrier_delay],
        "weather_delay": [weather_delay],
        "scheduled_departure_time": [time_to_minutes(departure_time)],
        "scheduled_arrival_time": [time_to_minutes(arrival_time)],
    }

    df_input = pd.DataFrame(input_data)
    df_encoded = pd.get_dummies(df_input)
    df_encoded = df_encoded.reindex(columns=model_columns, fill_value=0)

    try:
        prediction = model.predict(df_encoded)[0]
        st.success(f"🛬 Predicted Flight Status: **{prediction}**")
    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")
