import streamlit as st
import pandas as pd
import pickle
import datetime

# Load the model and its feature columns
@st.cache_data
def load_model():
    with open('model/flight_delay_model.pkl', 'rb') as f:
        model, feature_columns = pickle.load(f)
    return model, feature_columns

model, feature_columns = load_model()

# Load data to get dropdown options
df = pd.read_csv('final_airline_times_HHMM.csv')

st.title("✈️ Flight Delay Prediction App")
st.write("Enter flight details to predict if the flight will be delayed.")

# Input fields from the dataset
year = st.selectbox("Year", sorted(df['year'].dropna().unique()))
carrier = st.selectbox("Carrier", sorted(df['carrier'].dropna().unique()))
airport_name = st.selectbox("Airport Name", sorted(df['airport_name'].dropna().unique()))
scheduled_departure = st.time_input("Scheduled Departure Time", value=datetime.time(8, 0))
actual_arrival = st.time_input("Actual Arrival Time", value=datetime.time(10, 0))

# Process and Predict
if st.button("Predict Delay"):
    try:
        # Convert time to total minutes
        sched_dep_minutes = scheduled_departure.hour * 60 + scheduled_departure.minute
        actual_arr_minutes = actual_arrival.hour * 60 + actual_arrival.minute

        # Build input data
        input_dict = {
            'year': year,
            'carrier': carrier,
            'airport_name': airport_name,
            'scheduled_departure': sched_dep_minutes,
            'actual_arrival': actual_arr_minutes
        }
        input_df = pd.DataFrame([input_dict])

        # One-hot encode and align with training columns
        input_encoded = pd.get_dummies(input_df)
        missing_cols = set(feature_columns) - set(input_encoded.columns)
        for col in missing_cols:
            input_encoded[col] = 0
        input_encoded = input_encoded[feature_columns]  # Ensure column order

        # Prediction
        prediction = model.predict(input_encoded)[0]
        proba = model.predict_proba(input_encoded)[0][1]

        if prediction == 1:
            st.error(f"⚠️ Flight is likely to be **delayed** (probability: {proba:.2f})")
        else:
            st.success(f"✅ Flight is likely to be **on time** (probability: {1 - proba:.2f})")

    except Exception as e:
        st.warning(f"Prediction failed: {e}")
