import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# -------------------------------------------------------------------
# 1. CACHED RESOURCE LOADING
# -------------------------------------------------------------------
@st.cache_resource
def load_models():
    """Load all trained models and encoders."""
    models = {
        "workout_model": joblib.load("workout_type_model.pkl"),
        "workout_scaler": joblib.load("scaler_workout.pkl"),
        "workout_encoder": joblib.load("label_encoder_workout.pkl"),
        "freq_model": joblib.load("frequency_model.pkl"),
        "freq_scaler": joblib.load("scaler_freq.pkl"),
        "freq_encoder": joblib.load("label_encoder_freq.pkl"),
        "dur_model": joblib.load("duration_model.pkl"),
        "dur_scaler": joblib.load("scaler_freq.pkl"),  # same scaler as freq
        "cal_model": joblib.load("calorie_model.pkl"),
        "cal_scaler": joblib.load("calories_scaler.pkl"),
    }
    return models

@st.cache_data
def load_data():
    """Load and preprocess the food dataset."""
    df = pd.read_csv("final_ind.csv")
    df = remove_non_meal_foods(df)
    df["category"] = df["food"].apply(classify_food)
    df["health_score"] = df.apply(lambda row: compute_health_score(row, goal="maintenance"), axis=1)
    df = df.sort_values("health_score", ascending=False)
    return df

# -------------------------------------------------------------------
# 2. ALL HELPER FUNCTIONS (unchanged – keep your existing implementations)
# -------------------------------------------------------------------
def bmi_class(weight, height):
    bmi = weight / ((height/100) ** 2)
    if bmi < 18.5:
        return "Underweight"
    elif 18.5 <= bmi <= 24.9:
        return "Normal weight"
    elif 25 <= bmi <= 29.9:
        return "Overweight"
    else:
        return "Obese"

def remove_non_meal_foods(df):
    blacklist = [
        "salt", "pepper", "vinegar", "mustard", "spice", "seasoning", "extract",
        "essence", "vegetable oil", "coffee", "flavor", "stock", "sauce base",
        "burger", "pizza", "fries", "fried", "cake", "pastry", "chips", "cola",
        "soda", "chocolate", "ice cream", "candy", "biscuit", "donut", "noodle",
        "noodles", "instant noodles", "sausage", "salami", "nachos", "brownie",
        "milkshake", "muffin", "white bread", "cream", "mayonnaise", "sweet drink",
        "packaged juice", "cheese noodle ring", "bread sauce", "spinach burfi",
        "palak burfi", "sweet rice", "meethe chawal", "fruit punch","chutney",
        "Fish in coconut milk (Nariyal ke doodh ke saath machli)"
    ]
    pattern = r'\b(?:' + '|'.join(blacklist) + r')\b'
    df = df[~df["food"].str.lower().str.contains(pattern, na=False, regex=True)]
    df = df[df["Caloric Value"] > 20]
    return df

def classify_food(name):
    name = str(name).lower()
    if any(x in name for x in ["oil", "butter", "ghee", "margarine"]):
        return "fat"
    if any(x in name for x in ["chicken breast", "fish", "egg white", "tofu", "paneer"]):
        return "protein_lean"
    if any(x in name for x in ["chicken", "egg", "meat", "keema", "mutton"]):
        return "protein_fatty"
    if any(x in name for x in ["brown rice", "roti", "chapati", "oat", "quinoa", "poha", "dalia", "whole wheat"]):
        return "carb_complex"
    if any(x in name for x in ["white rice", "bread", "sugar", "sweet"]):
        return "carb_simple"
    if any(x in name for x in ["apple", "banana", "orange", "fruit", "berry", "mango", "grapes"]):
        return "fruit"
    if any(x in name for x in ["spinach", "broccoli", "carrot", "beans", "veg", "soup", "raita", "salad", "bhaji"]):
        return "vegetable"
    return "other"

def compute_health_score(row, goal="weight_loss"):
    protein = row.get("Protein", 0)
    fat = row.get("Fat", 0)
    sugar = row.get("Sugars", 0)
    fiber = row.get("Fiber", 0)
    if goal == "weight_loss":
        return protein * 3 - fat * 2.5 - sugar * 1.5 + fiber * 1
    elif goal == "muscle_gain":
        return protein * 4 - fat * 1.2 - sugar * 1 + fiber * 0.5
    else:  # maintenance
        return protein * 2 - fat * 1.5 - sugar * 1 + fiber * 0.8

def calorie_split(total):
    return {
        "breakfast": total * 0.20,
        "lunch":     total * 0.35,
        "dinner":    total * 0.30,
        "snacks":    total * 0.15
    }

def adjust_calories(base, goal):
    if goal == "weight_loss":
        return base * 0.8
    elif goal == "muscle_gain":
        return base * 1.15
    return base

def calculate_tdee(age, gender, weight_kg, height_cm, activity_level):
    if gender.lower() == 'male':
        bmr = 10 * weight_kg + 6.25 * height_cm - 5 * age + 5
    else:
        bmr = 10 * weight_kg + 6.25 * height_cm - 5 * age - 161
    multipliers = {
        'sedentary': 1.2,
        'light': 1.375,
        'moderate': 1.55,
        'active': 1.725,
        'very_active': 1.9
    }
    return bmr * multipliers.get(activity_level, 1.2)

def apply_diet_preference(df, preference):
    if preference and preference.lower() == "vegetarian":
        nonveg = ["chicken", "fish", "egg", "meat", "beef", "pork"]
        pattern = '|'.join(nonveg)
        return df[~df["food"].str.contains(pattern, case=False, na=False)]
    return df

def medical_filter(df, conditions):
    if not conditions:
        return df
    if isinstance(conditions, str):
        conditions = [conditions]
    conditions = [c.lower() for c in conditions]
    df = df.copy()
    for col in ["Sugars", "Fat", "Carbohydrates", "Protein", "Fiber"]:
        if col not in df.columns:
            df[col] = 0
    if "diabetes" in conditions:
        df = df[(df["Sugars"] < 8) & (df["Carbohydrates"] < 35)]
    if "heart" in conditions:
        df = df[df["Fat"] < 8]
    if "cholesterol" in conditions:
        df = df[df["Fat"] < 6]
    if "kidney" in conditions:
        df = df[df["Protein"] < 15]
    return df

def filter_diet(df, mode):
    # mode: 'normal', 'high_protein', 'low_carb'
    if mode == "high_protein":
        return df[df["Protein"] > 15]
    elif mode == "low_carb":
        return df[df["Carbohydrates"] < 20]
    return df

# Allowed categories per meal
meal_allowed_categories = {
    "breakfast": ["protein_lean", "protein_fatty", "carb_complex", "carb_simple", "fruit", "vegetable"],
    "lunch":     ["protein_lean", "protein_fatty", "carb_complex", "carb_simple", "vegetable", "fruit"],
    "dinner":    ["protein_lean", "protein_fatty", "carb_complex", "carb_simple", "vegetable", "fruit"],
    "snacks":    ["fruit", "vegetable", "protein_lean", "carb_complex"]
}

# ----- Include your full build_meal, diet_planner, generate_exercise_plan, recommend_yoga functions here -----
# (They are identical to your previous code – copy them in. For brevity I'm not repeating them, but you must.)
# Make sure to paste your existing implementations between these comments.

# -------------------------------------------------------------------
# 3. STREAMLIT UI – TWO‑COLUMN DASHBOARD
# -------------------------------------------------------------------
def main():
    st.set_page_config(page_title="Health & Fitness Recommendation System", layout="wide")
    
    # Initialize session state for plan results
    if "plan_generated" not in st.session_state:
        st.session_state.plan_generated = False

    # --- LEFT SIDEBAR (fixed navigation + inputs) ---
    with st.sidebar:
        st.markdown("## 🧭 Navigation")
        page = st.radio(
            "Go to",
            ["📌 About", "💪 Exercise Plan", "🧘 Yoga Plan", "🍽️ Diet Plan"],
            index=0,
            label_visibility="collapsed"
        )
        st.markdown("---")

        st.header("📋 Your Profile")
        age = st.number_input("Age", min_value=10, max_value=100, value=30, step=1)
        gender = st.selectbox("Gender", ["male", "female"])
        weight = st.number_input("Weight (kg)", min_value=30.0, max_value=200.0, value=70.0, step=0.1)
        height = st.number_input("Height (cm)", min_value=100.0, max_value=250.0, value=170.0, step=0.1)

        st.header("🎯 Lifestyle & Goals")
        activity_level = st.selectbox(
            "Activity Level",
            ["sedentary", "light", "moderate", "active", "very_active"]
        )
        goal_display = st.selectbox(
            "Primary Goal",
            ["Weight Loss", "Weight Gain", "Muscle Gain", "Maintenance"]
        )
        preference = st.selectbox(
            "Diet Preference",
            ["none", "vegetarian"]
        )
        experience = st.slider(
            "Exercise Experience (1 = beginner, 4 = expert)",
            min_value=1, max_value=4, value=2, step=1
        )
        mode_display = st.selectbox(
            "Diet Mode",
            ["Normal", "High Protein", "Low Carb"]
        )
        conditions = st.multiselect(
            "Medical Conditions (if any)",
            ["diabetes", "heart", "cholesterol", "kidney"]
        )

        generate = st.button("🚀 Generate My Plan", type="primary", use_container_width=True)

    # --- RIGHT MAIN PANEL (dynamic content) ---
    # Title always visible
    st.markdown("# 🏋️‍♂️ Health Fitness Recommendation System")

    if generate:
        with st.spinner("Creating your personalized plan..."):
            pref = None if preference == "none" else preference
            conds = conditions if conditions else None
            # Call your health_fitness_system function (must be defined)
            result = health_fitness_system(
                age=age,
                gender=gender,
                weight=weight,
                height=height,
                activity_level=activity_level,
                goal_display=goal_display,
                preference=pref,
                experience=experience,
                mode_display=mode_display,
                conditions=conds,
                df=load_data(),        # careful: load_data is cached, okay
                models=load_models()
            )
            st.session_state.result = result
            st.session_state.plan_generated = True
            st.session_state.goal_display = goal_display
            st.session_state.experience = experience
            st.success("Plan generated! You can now view it in the tabs.")

    # Display content based on selected page
    if page == "📌 About":
        st.header("📌 About the System")
        st.markdown("""
        **Welcome to the Personalized Health & Fitness Recommendation System!**  

        This AI‑powered tool creates custom‑tailored diet and exercise plans based on your unique profile.  
        - **Input** your age, gender, weight, height, activity level, and goals.  
        - The system uses **machine learning models** trained on real fitness data to predict the best workout type, frequency, duration, and calorie burn.  
        - It also generates a **balanced diet plan** with Indian and international foods, respecting your dietary preferences and medical conditions.  
        - Finally, it recommends **yoga poses** suitable for your experience level and fitness goal.

        **How to use:**  
        1. Fill in your details in the left sidebar.  
        2. Click **"Generate My Plan"**.  
        3. Navigate through the tabs above to see your personalized plans.

        *Stay healthy, stay fit!*  
        """)

    elif page == "💪 Exercise Plan":
        st.header("💪 Your Exercise Plans")
        if st.session_state.get("plan_generated", False):
            ex_plans = st.session_state.result["Exercise Plans"]
            cols = st.columns(len(ex_plans))
            for i, plan in enumerate(ex_plans):
                with cols[i]:
                    st.subheader(f"Option {i+1}")
                    st.metric("Workout Type", plan["Workout Type"])
                    st.metric("Frequency", f"{plan['Workout Frequency (days/week)']} days/week")
                    st.metric("Session Duration", f"{plan['Session Duration (minutes)']} min")
                    st.metric("Calories Burned", f"{plan['Estimated Calories Burned']} kcal")
        else:
            st.info("👈 Please generate a plan first using the sidebar.")

    elif page == "🧘 Yoga Plan":
        st.header("🧘 Yoga Recommendations")
        if st.session_state.get("plan_generated", False):
            goal_internal = ("weight_loss" if "Weight Loss" in st.session_state.goal_display
                             else "muscle_gain" if "Gain" in st.session_state.goal_display
                             else "maintenance")
            yoga_poses = recommend_yoga(st.session_state.experience, goal_internal, age)
            for pose in yoga_poses:
                st.markdown(f"**{pose['name']}**  \n{pose['desc']}")
            # Optional collage image
            collage_path = "images/yoga_collage.jpg"
            if os.path.exists(collage_path):
                st.image(collage_path, caption="Yoga Pose Collage", use_column_width=True)
        else:
            st.info("👈 Please generate a plan first using the sidebar.")

    elif page == "🍽️ Diet Plan":
        st.header("🍽️ Your Daily Diet Plan")
        if st.session_state.get("plan_generated", False):
            diet = st.session_state.result["Recommended Diet"]
            meals_order = ["breakfast", "lunch", "dinner", "snacks"]
            for meal in meals_order:
                df_meal = diet.get(meal)
                if df_meal is not None and not df_meal.empty:
                    with st.expander(f"**{meal.title()}**", expanded=True):
                        for _, row in df_meal.iterrows():
                            st.markdown(f"• **{row['food']}**")
                else:
                    with st.expander(f"**{meal.title()}**", expanded=False):
                        st.info("No foods selected for this meal.")
            # Daily totals
            non_empty = [diet[m] for m in meals_order if not diet[m].empty]
            if non_empty:
                all_meals = pd.concat(non_empty, ignore_index=True)
                total_cal = all_meals["Caloric Value"].sum()
                total_prot = all_meals["Protein"].sum()
                total_carb = all_meals["Carbohydrates"].sum()
                total_fat = all_meals["Fat"].sum()
                st.subheader("📊 Daily Totals")
                st.markdown(f"**Calories:** {total_cal:.0f} kcal  \n"
                            f"**Protein:** {total_prot:.1f}g  \n"
                            f"**Carbs:** {total_carb:.1f}g  \n"
                            f"**Fat:** {total_fat:.1f}g")
        else:
            st.info("👈 Please generate a plan first using the sidebar.")

if __name__ == "__main__":
    main()
