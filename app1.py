# # app1.py
# import streamlit as st
# import pandas as pd
# import joblib

# # Load the Logistic Regression model
# log_reg = joblib.load('logistic_regression_model.pkl')

# # Title and Description
# st.title("Diagnosis Analysis with Logistic Regression")
# st.write("This app uses a Logistic Regression model to predict diagnosis based on user input.")

# # User inputs
# Alvarado_Score = st.number_input("Alvarado Score")
# Paedriatic_Appendicitis_Score = st.number_input("Paediatric Appendicitis Score")
# WBC_Count = st.number_input("WBC Count")
# Neutrophilia = st.number_input("Neutrophilia")
# Management = st.number_input("Management")
# Severity = st.number_input("Severity")
# Length_of_Stay = st.number_input("Length of Stay")
# CRP = st.number_input("CRP")
# Peritonitis = st.number_input("Peritonitis")
# Age = st.number_input("Age")
# BMI = st.number_input("BMI")
# Height = st.number_input("Height")
# Weight = st.number_input("Weight")
# Sex = st.selectbox("Sex", options=[0, 1])  # Assuming 0 = Female, 1 = Male

# # Organize input into a single DataFrame row
# input_data = pd.DataFrame([[Alvarado_Score, Paedriatic_Appendicitis_Score, WBC_Count, Neutrophilia,
#                             Management, Severity, Length_of_Stay, CRP, Peritonitis, Age,
#                             BMI, Height, Weight, Sex]], 
#                           columns=['Alvarado_Score', 'Paedriatic_Appendicitis_Score', 'WBC_Count', 
#                                    'Neutrophilia', 'Management', 'Severity', 'Length_of_Stay', 
#                                    'CRP', 'Peritonitis', 'Age', 'BMI', 'Height', 'Weight', 'Sex'])

# # Predict button
# if st.button("Predict"):
#     # Make prediction
#     log_reg_pred = log_reg.predict(input_data)[0]
#     log_reg_accuracy = 0.85  # Placeholder; update based on actual accuracy

#     # Display the prediction and accuracy
#     st.subheader("Prediction")
#     st.write("Logistic Regression Prediction:", log_reg_pred)
#     st.write("Logistic Regression Accuracy:", log_reg_accuracy)


import streamlit as st
import pandas as pd
import joblib

# Load the Logistic Regression model
log_reg = joblib.load('logistic_regression_model.pkl')

# Title and Description
st.title("Diagnosis Analysis with Logistic Regression")
st.write("This app uses a Logistic Regression model to predict diagnosis based on user input or data from a CSV file.")

# Option for User Input (Manually)
st.header("Manual Input for Prediction")
# User inputs
Alvarado_Score = st.number_input("Alvarado Score")
Paedriatic_Appendicitis_Score = st.number_input("Paediatric Appendicitis Score")
WBC_Count = st.number_input("WBC Count")
Neutrophilia = st.number_input("Neutrophilia")
Management = st.number_input("Management")
Severity = st.number_input("Severity")
Length_of_Stay = st.number_input("Length of Stay")
CRP = st.number_input("CRP")
Peritonitis = st.number_input("Peritonitis")
Age = st.number_input("Age")
BMI = st.number_input("BMI")
Height = st.number_input("Height")
Weight = st.number_input("Weight")
Sex = st.selectbox("Sex", options=[0, 1])  # Assuming 0 = Female, 1 = Male

# Organize manual input data into a DataFrame
input_data_manual = pd.DataFrame([[Alvarado_Score, Paedriatic_Appendicitis_Score, WBC_Count, Neutrophilia,
                                   Management, Severity, Length_of_Stay, CRP, Peritonitis, Age,
                                   BMI, Height, Weight, Sex]], 
                                 columns=['Alvarado_Score', 'Paedriatic_Appendicitis_Score', 'WBC_Count', 
                                          'Neutrophilia', 'Management', 'Severity', 'Length_of_Stay', 
                                          'CRP', 'Peritonitis', 'Age', 'BMI', 'Height', 'Weight', 'Sex'])

# URL of the CSV file hosted on GitHub
csv_url = 'https://github.com/sarah-noorani34/Abdomen-cancer-ML/blob/main/top_10_features.csv' 
df_uploaded = pd.read_csv(csv_url)

# Manual Input Prediction Button
if st.button("Predict Input"):
    # Make prediction for manual input
    if not (Alvarado_Score and Paedriatic_Appendicitis_Score and WBC_Count and Neutrophilia and 
            Management and Severity and Length_of_Stay and CRP and Peritonitis and Age and BMI and 
            Height and Weight and Sex):
        st.error("Please fill all the fields for manual input.")
    else:
        log_reg_pred = log_reg.predict(input_data_manual)[0]
        log_reg_pred_proba = log_reg.predict_proba(input_data_manual)[0][1]  # Probability for class 1

        # Display the prediction and probability
        st.subheader("Prediction for Manual Input")
        st.write(f"Logistic Regression Prediction (0 = No, 1 = Yes): {log_reg_pred}")
        st.write(f"Probability of Diagnosis (Yes): {round(log_reg_pred_proba, 4)}")

        # Display model accuracy (if you have it)
        log_reg_accuracy = 0.85  # Placeholder accuracy, you can update it based on the actual model's performance
        st.write(f"Logistic Regression Model Accuracy: {log_reg_accuracy}")
