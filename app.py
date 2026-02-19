import streamlit as st

import streamlit as st
import base64

st.set_page_config(page_title="Health Recommendation System", layout="centered")

def set_bg(image_file):
    with open(image_file, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()
    
    page_bg_img = f"""
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{encoded}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }}
    </style>
    """
    st.markdown(page_bg_img, unsafe_allow_html=True)

set_bg("background.png.png")

st.title("Health Recommendation System")




age = st.text_input("Age (years)", placeholder="Enter your age")
weight = st.text_input("Weight (kg)", placeholder="Enter your weight")
height = st.text_input("Height (cm)", placeholder="Enter your height")

gender = st.selectbox(
    "Gender",
    ["Male", "Female", "Other"],
    index=None,
    placeholder="Select gender"
)

activity_level = st.selectbox(
    "Activity Level",
    ["Sedentary", "Light", "Moderate", "Active", "Very Active"],
    index=None,
    placeholder="Select activity level"
)

goal = st.selectbox(
    "Goal",
    ["Weight Gain", "Weight Loss", "Maintenance"],
    index=None,
    placeholder="Select goal"
)

preference = st.selectbox(
    "Food Preference",
    ["Vegetarian", "Non-Veg"],
    index=None,
    placeholder="Select food preference"
)

experience = st.selectbox(
    "Fitness Experience (years)",
    [1, 2, 3],
    index=None,
    placeholder="Select experience"
)

mode = st.selectbox(
    "Diet Mode",
    ["Normal", "Diabetic-Friendly", "High-Protein"],
    index=None,
    placeholder="Select diet mode"
)

conditions = st.multiselect(
    "Medical Conditions (Optional)",
    ["Diabetes", "Heart", "Cholesterol", "Kidney"]
)

if st.button("Submit"):
    if not (age and weight and height and gender and activity_level and goal and preference and experience and mode):
        st.warning("Please fill all required fields.")
    else:
        st.success("Details submitted successfully!")

    
                  



