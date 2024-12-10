import streamlit as st
import numpy as np
import requests

# Streamlit UI
st.title("Personalized Healthcare Recommendations")
st.subheader("Enter your health details:")

# Input fields
age = st.number_input('Age', min_value=18, max_value=120, value=30)
bmi = st.number_input('BMI', min_value=10, max_value=50, value=22)
cholesterol = st.number_input('Cholesterol Level', min_value=100, max_value=300, value=180)
glucose = st.number_input('Glucose Level', min_value=50, max_value=300, value=100)

# When the user clicks 'Generate Recommendation'
if st.button('Generate Recommendation'):
    user_data = {
        'features': [age, bmi, cholesterol, glucose]
    }

    # Send the data to the Flask API for prediction
    response = requests.post("http://127.0.0.1:5000/predict", json=user_data)
    
    # Get the prediction from the response and display it
    prediction = response.json()['prediction']
    st.write(f"Recommended healthcare plan: {prediction}")
