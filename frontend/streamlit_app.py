# AI helped generate this code.

# 1. Import packages
import requests
import streamlit as st


# 2. Set up the page and grade inputs
st.title("Student Outcome Predictor")
st.write("Enter two grades on the 0–200 scale to see the predicted student outcome.")

admission_grade = st.number_input(
    "Admission grade", min_value=0.0, max_value=200.0, value=150.0
)
previous_grade = st.number_input(
    "Previous qualification grade", min_value=0.0, max_value=200.0, value=145.0
)


# 3. Send the grades to the API when the user clicks Predict
if st.button("Predict"):
    grades = {
        "admission_grade": admission_grade,
        "previous_qualification_grade": previous_grade,
    }

    api_url = "https://student-api-786379901597.us-central1.run.app/predict"

    try:
        response = requests.post(api_url, json=grades, timeout=30)
        response.raise_for_status()
        prediction = response.json()["prediction"]
        st.success(f"Predicted outcome: {prediction}")
    except (requests.RequestException, ValueError, KeyError):
        st.error("Could not get a prediction from the API. Please try again.")