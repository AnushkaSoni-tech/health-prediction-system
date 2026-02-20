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
# 2. ALL HELPER FUNCTIONS (unchanged – insert your existing functions here)
# -------------------------------------------------------------------
# (Copy your existing helper functions: bmi_class, remove_non_meal_foods, classify_food,
# compute_health_score, calorie_split, adjust_calories, calculate_tdee, apply_diet_preference,
# medical_filter, filter_diet, meal_allowed_categories, build_meal, diet_planner,
# generate_exercise_plan, recommend_yoga, health_fitness_system)
# For brevity, I'm not repeating them. Make sure they are present.

# -------------------------------------------------------------------
# 3. STREAMLIT UI – TWO‑COLUMN DASHBOARD with input in About tab
# -------------------------------------------------------------------
def main():
    st.set_page_config(page_title="Health & Fitness Recommendation System", layout="wide")
    
    # Initialize session state for plan results and inputs
    if "plan_generated" not in st.session_state:
        st.session_state.plan_generated = False
    # We'll also store the user inputs to reuse them when generating
    if "user_inputs" not in st.session_state:
        st.session_state.user_inputs = {}

    # --- LEFT SIDEBAR (navigation only) ---
    with st.sidebar:
        st.markdown("## 🧭 Navigation")
        page = st.radio(
            "Go to",
            ["📌 About", "💪 Exercise Plan", "🧘 Yoga Plan", "🍽️ Diet Plan"],
            index=0,
            label_visibility="collapsed"
        )
        st.markdown("---")
        st.markdown("### ℹ️ Instructions\nFill your details in the **About** tab and generate your plan.")

    # --- RIGHT MAIN PANEL (dynamic content) ---
    st.markdown("# 🏋️‍♂️ Health Fitness Recommendation System")

    # If the page is About, show the input form and generation button
    if page == "📌 About":
        st.header("📌 About the System")
        st.markdown("""
        **Welcome to the Personalized Health & Fitness Recommendation System!**  

        This AI‑powered tool creates custom‑tailored diet and exercise plans based on your unique profile.  
        - **Input** your details below.  
        - The system uses **machine learning models** to predict the best workout and diet plans.  
        - After generating, you can view your plans in the other tabs.

        *Stay healthy, stay fit!*  
        """)

        # Input form
        with st.form("user_input_form"):
            col1, col2 = st.columns(2)
            with col1:
                age = st.number_input("Age", min_value=10, max_value=100, value=30, step=1)
                gender = st.selectbox("Gender", ["male", "female"])
                weight = st.number_input("Weight (kg)", min_value=30.0, max_value=200.0, value=70.0, step=0.1)
                height = st.number_input("Height (cm)", min_value=100.0, max_value=250.0, value=170.0, step=0.1)
            with col2:
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
            generate = st.form_submit_button("🚀 Generate My Plan", type="primary", use_container_width=True)

        if generate:
            with st.spinner("Creating your personalized plan..."):
                pref = None if preference == "none" else preference
                conds = conditions if conditions else None
                # Load data and models (cached)
                df = load_data()
                models = load_models()
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
                    df=df,
                    models=models
                )
                # Store everything needed for other tabs
                st.session_state.result = result
                st.session_state.plan_generated = True
                st.session_state.goal_display = goal_display
                st.session_state.experience = experience
                # Also store inputs to potentially reuse (optional)
                st.session_state.user_inputs = {
                    "age": age,
                    "gender": gender,
                    "weight": weight,
                    "height": height,
                    "activity_level": activity_level,
                    "goal_display": goal_display,
                    "preference": preference,
                    "experience": experience,
                    "mode_display": mode_display,
                    "conditions": conditions,
                }
                st.success("Plan generated! You can now view it in the other tabs.")

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
            st.info("👈 Please go to the **About** tab, fill in your details, and generate a plan first.")

    elif page == "🧘 Yoga Plan":
        st.header("🧘 Yoga Recommendations")
        if st.session_state.get("plan_generated", False):
            goal_internal = ("weight_loss" if "Weight Loss" in st.session_state.goal_display
                             else "muscle_gain" if "Gain" in st.session_state.goal_display
                             else "maintenance")
            yoga_poses = recommend_yoga(st.session_state.experience, goal_internal, st.session_state.user_inputs.get("age", 30))
            for pose in yoga_poses:
                st.markdown(f"**{pose['name']}**  \n{pose['desc']}")
            # Optional collage image
            collage_path = "images/yoga_collage.jpg"
            if os.path.exists(collage_path):
                st.image(collage_path, caption="Yoga Pose Collage", use_column_width=True)
        else:
            st.info("👈 Please go to the **About** tab, fill in your details, and generate a plan first.")

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
            st.info("👈 Please go to the **About** tab, fill in your details, and generate a plan first.")

if __name__ == "__main__":
    main()
