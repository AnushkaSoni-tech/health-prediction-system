import streamlit as st

st.set_page_config(page_title="Health Prediction System", page_icon="🩺")

st.title("Health Prediction System")

age = st.text_input("Age (years)")
weight = st.text_input("Weight (kg)")
height = st.text_input("Height (cm)")
gender = st.selectbox("Gender", ["Select", "Male", "Female"])

if st.button("Submit"):
    st.success("Details submitted successfully!")
    st.write("Age:", age)
    st.write("Weight:", weight)
    st.write("Height:", height)
    st.write("Gender:", gender)
