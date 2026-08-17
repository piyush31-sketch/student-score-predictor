import streamlit as st
import joblib
import pandas as pd

# Loading the trained model
model = joblib.load("model.joblib")

# Page configuration
st.set_page_config(
    page_title="Student Score Predictor",
    page_icon="🎓"
)

# Title
st.title("🎓 Student Score Predictor")

st.write(
    "Enter the student's details below to predict their score."
)

# User inputs
age = st.number_input(
    "Age",
    min_value=18,
    max_value=100,
    value=25
)

ai = st.selectbox(
    "Knows AI?",
    options=[0, 1],
    format_func=lambda x: "Yes" if x == 1 else "No"
)

ml = st.selectbox(
    "Knows ML?",
    options=[0, 1],
    format_func=lambda x: "Yes" if x == 1 else "No"
)

# Prediction button
if st.button("Predict Score"):

    new_student = pd.DataFrame({
        "Age": [age],
        "AI": [ai],
        "ML": [ml]
    })

    prediction = model.predict(new_student)

    st.success(f"Predicted Score: {prediction[0]:.2f}")