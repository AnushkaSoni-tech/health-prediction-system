import streamlit as st

st.set_page_config(page_title="Health Prediction System", page_icon="🩺", layout="centered")

st.title("Health Prediction System")
age = st.text_input("Age (years)")
weight = st.text_input("Weight (kg)")
height = st.text_input("Height (cm)")
age = int(age)
weight = float(weight)
height = float(height)
gender = st.selectbox("Gender", ["","Select", "Male", "Female"])
