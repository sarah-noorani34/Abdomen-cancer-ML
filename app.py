import streamlit as st
import joblib
import numpy as np

# Load the trained model
model = joblib.load('logistic_regression_model.pkl')  # Ensure 'model.pkl' is in the same directory or provide a full path

# Set up the title and instructions
st.title("Abdominal Cancer Diagnosis Prediction")
st.write("Enter the following medical details to predict the diagnosis.")

# Input fields for each feature
alvarado_score = st.number_input("Alvarado Score", min_value=0.0, max_value=10.0, step=0.1)
paediatric_appendicitis_score = st.number_input("Paediatric Appendicitis Score", min_value=0.0, max_value=10.0, step=0.1)
wbc_count = st.number_input("WBC Count", min_value=0.0, max_value=100.0, step=0.1)
neutrophilia = st.selectbox("Neutrophilia (1 if present, 0 if not)", [0, 1])
management = st.selectbox("Management (0-2)", [0, 1, 2])
severity = st.selectbox("Severity (1-3)", [1, 2, 3])
length_of_stay = st.number_input("Length of Stay (days)", min_value=0.0, max_value=100.0, step=0.1)
crp = st.number_input("CRP", min_value=0.0, max_value=100.0, step=0.1)
peritonitis = st.selectbox("Peritonitis (0-2)", [0, 1, 2])
age = st.number_input("Age", min_value=0.0, max_value=120.0, step=0.1)
bmi = st.number_input("BMI", min_value=0.0, max_value=60.0, step=0.1)
height = st.number_input("Height (cm)", min_value=0.0, max_value=250.0, step=0.1)
weight = st.number_input("Weight (kg)", min_value=0.0, max_value=200.0, step=0.1)
sex = st.selectbox("Sex (0 for male, 1 for female)", [0, 1])

# Prepare the input data for prediction
input_data = np.array([[alvarado_score, paediatric_appendicitis_score, wbc_count, neutrophilia,
                        management, severity, length_of_stay, crp, peritonitis,
                        age, bmi, height, weight, sex]])

# Predict button
if st.button("Predict Diagnosis"):
    prediction = model.predict(input_data)
    diagnosis = "Positive for Abdominal Cancer" if prediction[0] == 1 else "Negative for Abdominal Cancer"
    st.write(f"The predicted diagnosis is: **{diagnosis}**")