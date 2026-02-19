import streamlit as st

st.set_page_config(page_title="Health Prediction System", page_icon="🩺", layout="centered")

st.title("Health Prediction System")
st.write("Enter your details to calculate BMI and BMI Category.")

age = st.number_input("Age (years)", min_value=1, max_value=120, value=20)
weight = st.number_input("Weight (kg)", min_value=1.0, max_value=300.0, value=60.0, step=0.1)
height = st.number_input("Height (m)", min_value=0.5, max_value=2.5, value=1.70, step=0.01)

def bmi_category(bmi: float) -> str:
    if bmi < 18.5:
        return "Underweight"
    elif bmi < 25:
        return "Normal"
    elif bmi < 30:
        return "Overweight"
    else:
        return "Obese"

if st.button("Predict"):
    bmi = weight / (height ** 2)
    st.subheader(f"BMI: {bmi:.2f}")
    st.success(f"Category: {bmi_category(bmi)}")
