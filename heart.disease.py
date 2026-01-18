import streamlit as st
import pickle
import numpy as np

# Load model & scaler (simple way)
model = pickle.load(open("rf_classifier.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

st.title("Heart Disease Prediction")

gender = st.selectbox("Gender", ["Male", "Female"])
age = st.number_input("Age", 1, 120)

smoker = st.selectbox("Smoker", ["Yes", "No"])
cigs = st.number_input("Cigarettes per Day", 0)

bpmeds = st.selectbox("BP Medicines", ["Yes", "No"])
stroke = st.selectbox("Stroke History", ["Yes", "No"])
hypertension = st.selectbox("Hypertension", ["Yes", "No"])
diabetes = st.selectbox("Diabetes", ["Yes", "No"])

chol = st.number_input("Total Cholesterol")
sysbp = st.number_input("Systolic BP")
diabp = st.number_input("Diastolic BP")
bmi = st.number_input("BMI")
hr = st.number_input("Heart Rate")
glucose = st.number_input("Glucose")

if st.button("Predict"):

    data = np.array([[
        1 if gender == "Male" else 0,
        age,
        1 if smoker == "Yes" else 0,
        cigs,
        1 if bpmeds == "Yes" else 0,
        1 if stroke == "Yes" else 0,
        1 if hypertension == "Yes" else 0,
        1 if diabetes == "Yes" else 0,
        chol, sysbp, diabp, bmi, hr, glucose
    ]])

    data = scaler.transform(data)
    result = model.predict(data)

    if result[0] == 1:
        st.error("Patient has Heart Disease")
    else:
        st.success("Patient has No Heart Disease")