import streamlit as st
import joblib
import numpy as np

# Loading the XGBoost model
xgboost_model = joblib.load("C:\\Users\\Lenovo\\Downloads\\xgboost_model.pkl")

# Function to make predictions
def predict_calories(age, height, weight, duration, body_temp, heart_rate, gender):
    # Preparing the input array
    input_data = np.array([[age, height, weight, duration, body_temp, heart_rate, gender]])
    # Making prediction
    prediction = xgboost_model.predict(input_data)
    return prediction[0]

# Streamlit app
st.title("Calorie Burn Prediction")
st.write("This project aims to predict the number of calories burned during physical activities using XGBOOST.")
st.write("Enter your Details")
# User inputs
age = st.number_input("Age", min_value=0, max_value=120, value=25)
height = st.number_input("Height (cm)", min_value=50, max_value=250, value=170)
weight = st.number_input("Weight (kg)", min_value=20, max_value=200, value=70)
duration = st.number_input("Duration (minutes)", min_value=0, max_value=300, value=30)
body_temp = st.number_input("Body Temperature (Celsius)", min_value=30.0, max_value=45.0, value=37.0)
heart_rate = st.number_input("Heart Rate (beats per second)", min_value=30, max_value=200, value=70)
gender = st.selectbox("Gender", options=[1, 0], format_func=lambda x: 'Male' if x == 1 else 'Female')

# Predict button
if st.button("Predict"):
    prediction = predict_calories(age, height, weight, duration, body_temp, heart_rate, gender)
    st.success(f"Predicted Calorie Burn: {prediction:.2f} calories")


with st.container():
    with st.sidebar:
        members = [
            {"name": "Rohan Saraswat", "email": "rohan.saraswat2003@gmail. com", "linkedin": "https://www.linkedin.com/in/rohan-saraswat-a70a2b225/"},
        ]
        st.markdown("<h1 style='font-size:28px'>Developer</h1>", unsafe_allow_html=True)

        for member in members:
            st.write(f"Name: {member['name']}")
            st.write(f"Email: {member['email']}")
            st.write(f"LinkedIn: {member['linkedin']}")
            st.write("")
