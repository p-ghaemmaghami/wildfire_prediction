import streamlit as st
import joblib
import pandas as pd

# Load the trained model
model = joblib.load('tools/model.joblib')

# Streamlit app
# st.title("Wildfire Prediction")
st.markdown("<div style='text-align: center;'><h1>Wildfire Prediction</h1></div>", unsafe_allow_html=True)

# Input fields for user input
st.header("Input Features")
latitude = st.number_input("Latitude", value=45.0)
longitude = st.number_input("Longitude", value=-75.0)
year = st.number_input("Year", value=2024, min_value=1900, max_value=2100)  # Limit year input
month = st.number_input("Month", min_value=1, max_value=12)
day = st.number_input("Day", min_value=1, max_value=31)

# Create input DataFrame
input_data = pd.DataFrame({
    'LATITUDE': [latitude],
    'LONGITUDE': [longitude],
    'YEAR': [year],
    'MONTH': [month],
    'DAY': [day]
})

if st.button("Predict"):
    prediction = model.predict(input_data)
    print(prediction)
    if prediction[0] == 'no':
        st.write("✅ **No Fire**")
    else:
        st.write("<span style='color: red;'>❌ **Fire**</span>", unsafe_allow_html=True)
