import pandas as pd
import joblib

def predict_flight_delay(input_data):
    model = joblib.load("flight_delay_model.pkl")
    model_columns = joblib.load("feature_columns.pkl")
    input_df = pd.DataFrame(input_data)
    input_encoded = pd.get_dummies(input_df)
    input_encoded = input_encoded.reindex(columns=model_columns, fill_value=0)
    prediction = model.predict(input_encoded)
    return prediction[0]
