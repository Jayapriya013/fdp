import streamlit as st
import pandas as pd
import joblib

# Load model and encoders
model = joblib.load('flight_delay_model.pkl')
target_le = joblib.load('flight_status_encoder.pkl')
feature_encoders = joblib.load('feature_encoders.pkl')
trained_features_cols = joblib.load('feature_columns.pkl')

# Load dataset and clean column names
df = pd.read_csv('final_airline_times_HHMM.csv')
df.columns = df.columns.str.strip()

# Show column names for debugging
st.write("✅ Available Columns:", df.columns.tolist())

# Check that required columns exist
required_cols = ['airport_name', 'carrier', 'weather_delay']
missing_cols = [col for col in required_cols if col not in df.columns]
if missing_cols:
    st.error(f"Required columns missing from CSV: {missing_cols}")
    st.stop()

# Convert HH:MM time to minutes
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
    airport_name = st.selectbox("Airport Name", sorted(df['airport_name'].dropna().unique()))
    carrier = st.selectbox("Carrier", sorted(df['carrier'].dropna().unique()))
    weather_delay = st.number_input("Weather Delay (in minutes)", min_value=0, step=1)
    scheduled_departure_time = st.text_input("Scheduled Departure Time (HH:MM)", "10:00")
    scheduled_arrival_time = st.text_input("Scheduled Arrival Time (HH:MM)", "12:00")
    submit = st.form_submit_button("Predict Delay Status")

if submit:
    input_data = {
        'airport_name': [airport_name],
        'carrier': [carrier],
        'weather_delay': [weather_delay],
        'scheduled_departure_time': [time_to_minutes(scheduled_departure_time)],
        'scheduled_arrival_time': [time_to_minutes(scheduled_arrival_time)],
    }

    df_input = pd.DataFrame(input_data)

    # Encode categorical features
    for col in df_input.select_dtypes(include='object').columns:
        if col in feature_encoders:
            encoder = feature_encoders[col]
            def encode_value(val):
                val_str = str(val)
                if val_str in encoder.classes_:
                    return encoder.transform([val_str])[0]
                else:
                    st.warning(f"Unknown value '{val_str}' for '{col}'. Using 0.")
                    return 0
            df_input[col] = df_input[col].apply(encode_value)

    # Add any missing columns with zero and reorder
    for col in trained_features_cols:
        if col not in df_input.columns:
            df_input[col] = 0
    df_input = df_input[trained_features_cols]

    try:
        prediction = model.predict(df_input)
        result = target_le.inverse_transform(prediction)[0]
        st.success(f"🛬 Predicted Flight Status: **{result}**")
    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")
