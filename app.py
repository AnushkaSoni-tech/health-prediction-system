import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ---------------- BMI ----------------
def bmi_class(weight,height):
    bmi = weight / ((height/100) ** 2)
    if bmi<18.5:
        return "Underweight"
    elif bmi<=24.9:
        return "Normal weight"
    elif bmi<=29.9:
        return "Overweight"
    else:
        return "Obese"

# ---------------- CLEAN DATA ----------------
def remove_non_meal_foods(df):
    blacklist = [
        "salt","pepper","vinegar","mustard","spice","seasoning","extract",
        "essence","vegetable oil","coffee","flavor","stock","sauce base",
        "burger","pizza","fries","fried","cake","pastry","chips","cola",
        "soda","chocolate","ice cream","candy","biscuit"
    ]
    pattern='|'.join(blacklist)
    return df[~df["food"].str.contains(pattern, case=False, na=False)]

# ---------------- TDEE ----------------
def calculate_tdee(age, gender, weight_kg, height_cm, activity_level):

    if gender=='Male':
        bmr = 10*weight_kg + 6.25*height_cm - 5*age + 5
    else:
        bmr = 10*weight_kg + 6.25*height_cm - 5*age - 161

    multipliers={
        "sedentary":1.2,
        "light":1.375,
        "moderate":1.55,
        "active":1.725,
        "very_active":1.9
    }

    return bmr*multipliers[activity_level]

# ---------------- EXERCISE MODEL ----------------
def generate_exercise_plan(user_input):

    workout_model = joblib.load("workout_type_model.pkl")
    workout_scaler = joblib.load("scaler_workout.pkl")
    workout_encoder = joblib.load("label_encoder_workout.pkl")

    freq_model = joblib.load("frequency_model.pkl")
    freq_scaler = joblib.load("scaler_freq.pkl")
    freq_encoder = joblib.load("label_encoder_freq.pkl")

    arr = np.array([list(user_input.values())])

    w_scaled = workout_scaler.transform(arr)
    f_scaled = freq_scaler.transform(arr)

    workout = workout_encoder.inverse_transform(
        workout_model.predict(w_scaled)
    )[0]

    freq = freq_encoder.inverse_transform(
        freq_model.predict(f_scaled)
    )[0]

    return {
        "Workout Type": workout,
        "Workout Frequency": freq
    }

# ---------------- MAIN SYSTEM ----------------
def health_fitness_system(age, gender, weight, height, activity_level, goal):

    df = pd.read_csv("final_dataset.csv")
    df = remove_non_meal_foods(df)

    bmi = bmi_class(weight,height)
    calories = calculate_tdee(age, gender, weight, height, activity_level)

    # goal adjust
    if goal=="Weight Loss":
        calories*=0.8
    elif goal=="Muscle Gain":
        calories*=1.15

    # simple meal selection (top calories match)
    df = df.sort_values("Caloric Value")

    breakfast=df.sample(3)
    lunch=df.sample(3)
    dinner=df.sample(3)

    exercise=generate_exercise_plan({
        "age":age,
        "weight":weight,
        "height":height,
        "activity": ["sedentary","light","moderate","active","very_active"].index(activity_level)
    })

    return {
        "BMI":bmi,
        "Calories":int(calories),
        "Exercise":exercise,
        "Meals":{
            "Breakfast":breakfast,
            "Lunch":lunch,
            "Dinner":dinner
        }
    }

# ---------------- STREAMLIT UI ----------------
st.title("AI Health & Fitness Planner")

st.sidebar.header("Enter Details")

age = st.sidebar.slider("Age",10,80,25)
gender = st.sidebar.selectbox("Gender",["Male","Female"])
weight = st.sidebar.number_input("Weight (kg)",30,150,60)
height = st.sidebar.number_input("Height (cm)",120,210,170)

activity = st.sidebar.selectbox(
    "Activity Level",
    ["sedentary","light","moderate","active","very_active"]
)

goal = st.sidebar.selectbox(
    "Goal",
    ["Maintenance","Weight Loss","Muscle Gain"]
)

if st.sidebar.button("Generate Plan"):

    result = health_fitness_system(age,gender,weight,height,activity,goal)

    st.subheader("BMI Category")
    st.success(result["BMI"])

    st.subheader("Daily Calories")
    st.info(result["Calories"])

    st.subheader("Exercise Plan")
    st.write(result["Exercise"])

    st.subheader("Meal Plan")

    for meal,data in result["Meals"].items():
        st.write(f"### {meal}")
        st.dataframe(data)

