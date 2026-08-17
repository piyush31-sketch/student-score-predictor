import streamlit as st
import joblib
import pandas as pd

# Load the trained model
model = joblib.load("model.joblib")

# Page configuration
st.set_page_config(
    page_title="Student Score Predictor",
    page_icon="🎓",
    layout="centered"
)

# Title
st.title("🎓 Student Score Predictor")

st.write(
    "Predict a student's score using their age, AI knowledge, and ML knowledge."
)

st.divider()

# Input section
st.subheader("Student Information")

age = st.number_input(
    "Age",
    min_value=20,
    max_value=29,
    value=25,
    step=1
)

col1, col2 = st.columns(2)

with col1:
    ai = st.selectbox(
        "AI Knowledge",
        options=[0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )

with col2:
    ml = st.selectbox(
        "ML Knowledge",
        options=[0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )

st.divider()

# Prediction
if st.button("Predict Score", use_container_width=True):

    new_student = pd.DataFrame({
        "Age": [age],
        "AI": [ai],
        "ML": [ml]
    })

    prediction = model.predict(new_student)[0]

    st.success("Prediction generated successfully!")

    st.metric(
        label="Predicted Score",
        value=f"{prediction:.2f}"
    )

    if prediction >= 90:
        st.info("Excellent predicted performance! 🎯")
    elif prediction >= 75:
        st.info("Good predicted performance! 👍")
    else:
        st.info("There may be room for improvement. 📚")

st.divider()

# Model information
st.subheader("About the Model")

st.write(
    "This application uses a Linear Regression model trained on "
    "student age, AI knowledge, and ML knowledge."
)

st.write("Model evaluation:")

st.metric(
    label="Mean Absolute Error (MAE)",
    value="0.68"
)