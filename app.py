import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder
import datetime

# Load the trained model
model = pickle.load(open('model.pkl', 'rb'))

# Load and preprocess the dataset
df = pd.read_csv('Data/Processed_data15.csv')

# Label encoding setup
le_carrier = LabelEncoder()
df['carrier'] = le_carrier.fit_transform(df['carrier'])

le_origin = LabelEncoder()
df['origin'] = le_origin.fit_transform(df['origin'])

# Streamlit UI
st.title("✈️ Flight Delay Prediction App")

# Input fields
year = st.selectbox("Select Year", sorted(df['year'].unique()))
carrier = st.selectbox("Select Carrier", le_carrier.classes_)
airport_name = st.selectbox("Select Airport Name", le_origin.classes_)
scheduled_departure = st.time_input("Scheduled Departure Time (HH:MM)", value=datetime.time(8, 0))
actual_arrival = st.time_input("Actual Arrival Time (HH:MM)", value=datetime.time(10, 0))

# Predict button
if st.button("Predict Delay"):

    try:
        # Convert time to minutes from midnight
        sched_dep_minutes = scheduled_departure.hour * 60 + scheduled_departure.minute
        actual_arr_minutes = actual_arrival.hour * 60 + actual_arrival.minute

        # Prepare DataFrame with these new features (adjust based on model training)
        input_data = pd.DataFrame([[
            year,
            le_carrier.transform([carrier])[0],
            le_origin.transform([airport_name])[0],
            sched_dep_minutes,
            actual_arr_minutes
        ]], columns=['year', 'carrier', 'origin', 'sched_dep', 'arr_time'])

        # Predict
        prediction = model.predict(input_data)[0]

        if prediction == 1:
            st.error("🚨 The flight is likely to be **Delayed**.")
        else:
            st.success("✅ The flight is likely to be **On Time**.")

    except Exception as e:
        st.warning(f"Prediction failed: {e}")

