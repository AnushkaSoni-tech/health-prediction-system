import streamlit as st

st.set_page_config(page_title="Health Prediction System", page_icon="🩺", layout="centered")

st.title("Health Prediction System")
age = st.number_input("Age (years)", min_value=1, max_value=0, value=20)
weight = st.number_input("Weight (kg)", min_value=1.0, max_value=300.0, value=0, step=0.1)
height = st.number_input("Height (m)", min_value=0.5, max_value=2.5, value=0, step=0.01)
gender=st.selectbox("Gender",["Male","Female","Other"])
