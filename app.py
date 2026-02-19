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


def classify_food(food):
    food = str(food).lower()

    if any(x in food for x in ["chicken","egg","fish","meat","paneer","dal"]):
        return "protein"

    elif any(x in food for x in ["rice","roti","bread","oats","poha","upma"]):
        return "carb"

    elif any(x in food for x in ["salad","spinach","broccoli","vegetable","sabzi"]):
        return "vegetable"

    elif any(x in food for x in ["apple","banana","fruit","orange","papaya"]):
        return "fruit"

    elif any(x in food for x in ["oil","butter","ghee","cheese","nuts"]):
        return "fat"

    else:
        return "other"


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

def compute_health_score(row, goal="maintenance"):
    score = 0

    protein = row.get("Protein",0)
    fat = row.get("Fat",0)
    carbs = row.get("Carbohydrates",0)
    sugar = row.get("Sugars",0)
    calories = row.get("Caloric Value",0)

    # base scoring
    score += protein * 2
    score -= fat * 0.5
    score -= sugar * 1.5

    # goal adjustments
    if goal == "weight_loss":
        score -= calories * 0.02
    elif goal == "weight_gain":
        score += calories * 0.02

    return score


# ---------------- EXERCISE MODEL ----------------
def generate_exercise_plan(user_input):

    workout_model = joblib.load("workout_type_model.pkl")
    workout_scaler = joblib.load("scaler_workout.pkl")
    workout_encoder = joblib.load("label_encoder_workout.pkl")

    freq_model = joblib.load("frequency_model.pkl")
    freq_scaler = joblib.load("scaler_freq.pkl")
    freq_encoder = joblib.load("label_encoder_freq.pkl")

    # encode categorical variables SAME AS TRAINING
    gender = 1 if user_input["gender"].lower()=="male" else 0

    activity_map = {
        "sedentary":0,
        "light":1,
        "moderate":2,
        "active":3,
        "very_active":4
    }

    goal_map = {
        "weight_loss":0,
        "maintenance":1,
        "weight_gain":2
    }

    pref_map = {
        "veg":0,
        "nonveg":1
    }

    arr = np.array([[
        user_input["age"],
        gender,
        user_input["weight"],
        user_input["height"],
        activity_map[user_input["activity_level"]],
        goal_map[user_input["goal"]],
        pref_map[user_input["preference"]],
        user_input["experience"]
    ]])

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
def health_fitness_system(age, gender, weight, height, activity_level, goal,
                          preference=None, experience=1, mode="normal", conditions=None):

    import pandas as pd
    import os

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    df = pd.read_csv(os.path.join(BASE_DIR,"final_dataset.csv"))

    df = remove_non_meal_foods(df)

    df["category"] = df["food"].apply(classify_food)

    df["health_score"] = df.apply(
        lambda row: compute_health_score(row, goal=goal),
        axis=1
    )

    df = df.sort_values("health_score", ascending=False)

    bmi_value = weight / ((height/100) ** 2)
    bmi_category = bmi_class(weight, height)

    exercise_plan = generate_exercise_plan({
        "age": age,
        "gender": gender,
        "weight": weight,
        "height": height,
        "activity_level": activity_level,
        "goal": goal,
        "preference": preference,
        "experience": experience
    })

    tdee = calculate_tdee(age, gender, weight, height, activity_level)
    daily_cal = adjust_calories(tdee, goal)

    diet_plan = diet_planner(
        df=df,
        daily_cal=daily_cal,
        mode=mode,
        conditions=conditions,
        goal=goal,
        dietary_preference=preference
    )

    return {
        "BMI Class": bmi_category,
        "Exercise Plan": exercise_plan,
        "Recommended Diet": diet_plan
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




