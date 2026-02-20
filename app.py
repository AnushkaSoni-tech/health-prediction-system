import streamlit as st
import pandas as pd
import numpy as np
import joblib

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
    df = pd.read_csv("diet_friendly_foods.csv")
    df = remove_non_meal_foods(df)
    df["category"] = df["food"].apply(classify_food)
    df["health_score"] = df.apply(lambda row: compute_health_score(row, goal="maintenance"), axis=1)
    df = df.sort_values("health_score", ascending=False)
    return df

# -------------------------------------------------------------------
# 2. ALL HELPER FUNCTIONS
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
    if mode == "diabetic":
        return df[df["Sugars"] < 8]
    elif mode == "high_protein":
        return df[df["Protein"] > 15]
    return df

# Expanded allowed categories to include all relevant types for each meal
meal_allowed_categories = {
    "breakfast": ["protein_lean", "protein_fatty", "carb_complex", "carb_simple", "fruit", "vegetable"],
    "lunch":     ["protein_lean", "protein_fatty", "carb_complex", "carb_simple", "vegetable", "fruit"],
    "dinner":    ["protein_lean", "protein_fatty", "carb_complex", "carb_simple", "vegetable", "fruit"],
    "snacks":    ["fruit", "vegetable", "protein_lean", "carb_complex"]
}

def build_meal(df, target_cal, used_foods, meal_name, fat_cap_per_meal=None, goal="weight_loss"):
    """
    Builds one meal, aiming to hit the target calories while respecting the fat cap.
    Returns a DataFrame of selected foods. If normal selection fails, falls back to
    picking highest‑scoring items from allowed categories.
    """
    meal_items = []
    total_cal = 0.0
    total_fat = 0.0

    # Filter to allowed categories for this meal
    allowed = df[df["category"].isin(meal_allowed_categories[meal_name])].copy()
    available = allowed[~allowed["food"].isin(used_foods)].copy()
    if available.empty:
        return pd.DataFrame()

    # Soft macro targets (for scoring)
    if goal == "weight_loss":
        protein_target = target_cal * 0.30 / 4
        fat_target     = target_cal * 0.25 / 9
        carb_target    = target_cal * 0.45 / 4
    elif goal == "muscle_gain":
        protein_target = target_cal * 0.35 / 4
        fat_target     = target_cal * 0.30 / 9
        carb_target    = target_cal * 0.35 / 4
    else:  # maintenance
        protein_target = target_cal * 0.20 / 4
        fat_target     = target_cal * 0.30 / 9
        carb_target    = target_cal * 0.50 / 4

    def would_exceed_fat_cap(food):
        if fat_cap_per_meal and (total_fat + food["Fat"] > fat_cap_per_meal):
            return True
        return False

    # Category order based on meal type
    if meal_name == "breakfast":
        # Breakfast: carbs first, then fruits, then proteins
        category_order = ["carb_complex", "carb_simple", "fruit", "protein_lean", "protein_fatty", "vegetable", "other"]
    elif meal_name in ["lunch", "dinner"]:
        # Lunch/dinner: protein first, then carbs, then vegetables, then fruits
        category_order = ["protein_lean", "protein_fatty", "carb_complex", "carb_simple", "vegetable", "fruit", "other"]
    else:  # snacks
        category_order = ["fruit", "vegetable", "protein_lean", "carb_complex", "carb_simple", "other"]

    # ---- Phase 1: Pick one item from each priority category until we have a base ----
    for cat in category_order:
        if cat not in meal_allowed_categories[meal_name]:
            continue
        cat_df = available[available["category"] == cat]
        if cat_df.empty:
            continue

        # Filter based on ideal calorie range
        if cat.startswith("protein"):
            ideal_cal = target_cal * 0.35
            cat_df = cat_df[(cat_df["Caloric Value"] > 50) & (cat_df["Caloric Value"] < 400)]
        elif cat.startswith("carb"):
            ideal_cal = target_cal * 0.4
            cat_df = cat_df[cat_df["Caloric Value"] < 400]
        elif cat == "vegetable":
            ideal_cal = target_cal * 0.15
            cat_df = cat_df[cat_df["Caloric Value"] < 200]
        elif cat == "fruit":
            ideal_cal = target_cal * 0.15
            cat_df = cat_df[cat_df["Caloric Value"] < 200]
        else:
            ideal_cal = target_cal * 0.2
            cat_df = cat_df[cat_df["Caloric Value"] < 300]

        if cat_df.empty:
            continue

        # Score: combination of closeness to ideal calories and health score
        cat_df = cat_df.copy()
        cat_df["cal_diff"] = abs(cat_df["Caloric Value"] - ideal_cal)
        # Normalise health score to 0-1 range (approx)
        max_hs = cat_df["health_score"].max()
        min_hs = cat_df["health_score"].min()
        if max_hs > min_hs:
            cat_df["hs_norm"] = (cat_df["health_score"] - min_hs) / (max_hs - min_hs)
        else:
            cat_df["hs_norm"] = 0.5
        # Combine (lower cal_diff is better, higher hs_norm is better)
        # We'll use a weighted rank: 70% weight on cal_diff, 30% on health score
        cat_df["combined_score"] = 0.7 * (cat_df["cal_diff"] / cat_df["cal_diff"].max()) - 0.3 * cat_df["hs_norm"]
        best = cat_df.nsmallest(3, "combined_score").sample(1).iloc[0]

        # Check fat cap
        if would_exceed_fat_cap(best):
            continue

        # Add the food
        meal_items.append(best)
        total_cal += best["Caloric Value"]
        total_fat += best["Fat"]
        used_foods.add(best["food"])
        available = available[available["food"] != best["food"]]

        # If we already have a good base, we can break out of the priority loop
        # But we want at least 2 items for breakfast, 3 for lunch/dinner
        min_items = 2 if meal_name == "breakfast" else 3 if meal_name in ["lunch","dinner"] else 2
        if len(meal_items) >= min_items and total_cal >= target_cal * 0.5:
            break

    # ---- Phase 2: Ensure lunch/dinner have at least one protein and one carb ----
    if meal_name in ["lunch", "dinner"]:
        has_protein = any("protein" in item["category"] for item in meal_items)
        has_carb = any("carb" in item["category"] for item in meal_items)
        if not has_protein:
            # Force add a protein from remaining available
            protein_df = available[available["category"].str.contains("protein", na=False)]
            if not protein_df.empty:
                best_protein = protein_df.nlargest(1, "health_score").iloc[0]
                if not would_exceed_fat_cap(best_protein):
                    meal_items.append(best_protein)
                    total_cal += best_protein["Caloric Value"]
                    total_fat += best_protein["Fat"]
                    used_foods.add(best_protein["food"])
                    available = available[available["food"] != best_protein["food"]]
        if not has_carb:
            carb_df = available[available["category"].str.contains("carb", na=False)]
            if not carb_df.empty:
                best_carb = carb_df.nlargest(1, "health_score").iloc[0]
                if not would_exceed_fat_cap(best_carb):
                    meal_items.append(best_carb)
                    total_cal += best_carb["Caloric Value"]
                    total_fat += best_carb["Fat"]
                    used_foods.add(best_carb["food"])
                    available = available[available["food"] != best_carb["food"]]

    # ---- Phase 3: Fill remaining calories aggressively ----
    remaining_needed = target_cal - total_cal
    filler_candidates = available[~available["food"].isin(used_foods)].copy()
    # Exclude very large items (> 60% of remaining target) to avoid overshooting too much
    filler_candidates = filler_candidates[filler_candidates["Caloric Value"] <= remaining_needed * 1.2]
    # Sort by health score descending
    filler_candidates = filler_candidates.sort_values("health_score", ascending=False)

    for _, row in filler_candidates.iterrows():
        if total_cal >= target_cal * 0.95:
            break
        if row["food"] in used_foods:
            continue
        if would_exceed_fat_cap(row):
            continue
        meal_items.append(row)
        total_cal += row["Caloric Value"]
        total_fat += row["Fat"]
        used_foods.add(row["food"])

    # ---- Final validation ----
    # If meal is still empty or doesn't meet minimum criteria, fall back to a simple selection
    if len(meal_items) == 0 or (meal_name in ["lunch", "dinner"] and not any("protein" in item["category"] for item in meal_items)):
        # Fallback: pick top health score items from allowed categories, ignoring calorie and fat caps
        fallback_df = allowed[~allowed["food"].isin(used_foods)].copy()
        fallback_df = fallback_df.sort_values("health_score", ascending=False)
        meal_items = []
        total_cal = 0.0
        for _, row in fallback_df.iterrows():
            if len(meal_items) >= 3:
                break
            meal_items.append(row)
            used_foods.add(row["food"])
            total_cal += row["Caloric Value"]
        return pd.DataFrame(meal_items)

    if total_cal < target_cal * 0.4 and len(meal_items) < 2:
        # Too little food – fallback as above
        fallback_df = allowed[~allowed["food"].isin(used_foods)].copy()
        fallback_df = fallback_df.sort_values("health_score", ascending=False)
        meal_items = []
        total_cal = 0.0
        for _, row in fallback_df.iterrows():
            if len(meal_items) >= 3:
                break
            meal_items.append(row)
            used_foods.add(row["food"])
            total_cal += row["Caloric Value"]
        return pd.DataFrame(meal_items)

    return pd.DataFrame(meal_items)

def diet_planner(df, daily_cal, activity_level=None,
                 mode="normal", conditions=None,
                 goal="maintenance", preference=None):
    df = df.copy()
    # Recompute health score with the actual goal
    df["health_score"] = df.apply(lambda row: compute_health_score(row, goal=goal), axis=1)

    if preference:
        df = apply_diet_preference(df, preference)
    if conditions:
        df = medical_filter(df, conditions)
    df = filter_diet(df, mode)

    # Fat cap based on goal
    if goal == "weight_loss":
        fat_pct = 0.25
    elif goal == "muscle_gain":
        fat_pct = 0.30
    else:
        fat_pct = 0.30
    fat_cap_daily = (daily_cal * fat_pct) / 9

    splits = calorie_split(daily_cal)
    fat_cap_per_meal = {meal: (cal / daily_cal) * fat_cap_daily for meal, cal in splits.items()}

    used_foods = set()
    plan = {}
    for meal, cal in splits.items():
        meal_df = build_meal(df, cal, used_foods, meal,
                             fat_cap_per_meal.get(meal), goal=goal)
        plan[meal] = meal_df

    return plan

def generate_exercise_plan(user_input, models):
    """user_input = [age, gender_num, bmi, experience]"""
    # Unpack models
    workout_model = models["workout_model"]
    workout_scaler = models["workout_scaler"]
    workout_encoder = models["workout_encoder"]
    freq_model = models["freq_model"]
    freq_scaler = models["freq_scaler"]
    freq_encoder = models["freq_encoder"]
    dur_model = models["dur_model"]
    dur_scaler = models["dur_scaler"]
    cal_model = models["cal_model"]
    cal_scaler = models["cal_scaler"]

    # 1. Workout type
    workout_input = [user_input[:4]]  # Age, Gender, BMI, Experience
    workout_input_scaled = workout_scaler.transform(workout_input)
    workout_pred = workout_model.predict(workout_input_scaled)
    workout_type = workout_encoder.inverse_transform(workout_pred)[0]

    # 2. Frequency
    encoded_workout = freq_encoder.transform([workout_type])[0]
    freq_features = [
        user_input[0],   # Age
        user_input[1],   # Gender
        user_input[3],   # Experience
        user_input[2],   # BMI
        encoded_workout
    ]
    freq_input = freq_scaler.transform([freq_features])
    freq = int(freq_model.predict(freq_input)[0])

    # 3. Duration
    dur_features = [
        user_input[0],
        user_input[1],
        user_input[3],
        user_input[2],
        encoded_workout
    ]
    dur_input = dur_scaler.transform([dur_features])
    duration_hours = float(dur_model.predict(dur_input)[0])
    duration_minutes = round(duration_hours * 60)
    duration_minutes = max(15, min(duration_minutes, 180))

    # 4. Calories
    calorie_features = [
        user_input[0],
        user_input[1],
        user_input[3],
        user_input[2],
        encoded_workout,
        duration_hours
    ]
    calorie_scaled = cal_scaler.transform([calorie_features])
    calories = float(cal_model.predict(calorie_scaled)[0])
    calories = round(max(50, min(calories, 1200)))

    return {
        "Workout Type": workout_type,
        "Workout Frequency (days/week)": freq,
        "Session Duration (minutes)": duration_minutes,
        "Estimated Calories Burned": calories
    }

def health_fitness_system(age, gender, weight, height, activity_level, goal,
                          preference, experience, mode, conditions, df, models):
    gender_num = 1 if gender.lower() == 'male' else 0
    bmi_value = weight / ((height/100) ** 2)
    bmi_category = bmi_class(weight, height)

    # Exercise plan
    exercise_plan = generate_exercise_plan([age, gender_num, bmi_value, experience], models)

    # Diet plan
    tdee = calculate_tdee(age, gender, weight, height, activity_level)
    daily_cal = adjust_calories(tdee, goal)
    diet_plan = diet_planner(
        df=df,
        daily_cal=daily_cal,
        mode=mode,
        conditions=conditions,
        goal=goal,
        preference=preference
    )

    return {
        "BMI Class": bmi_category,
        "Exercise Plan": exercise_plan,
        "Recommended Diet": diet_plan
    }

# -------------------------------------------------------------------
# 3. STREAMLIT UI
# -------------------------------------------------------------------
def main():
    st.set_page_config(page_title="Personal Health & Fitness System", layout="wide")
    st.title("🏋️‍♂️ Personalized Health & Fitness Planner")
    st.markdown("Enter your details below to receive a customized diet and exercise plan.")

    # Load data and models (cached)
    with st.spinner("Loading models and food database..."):
        df = load_data()
        models = load_models()

    # Sidebar inputs
    with st.sidebar:
        st.header("Your Profile")
        age = st.number_input("Age", min_value=10, max_value=100, value=30, step=1)
        gender = st.selectbox("Gender", ["male", "female"])
        weight = st.number_input("Weight (kg)", min_value=30.0, max_value=200.0, value=70.0, step=0.1)
        height = st.number_input("Height (cm)", min_value=100.0, max_value=250.0, value=170.0, step=0.1)

        st.header("Lifestyle & Goals")
        activity_level = st.selectbox(
            "Activity Level",
            ["sedentary", "light", "moderate", "active", "very_active"]
        )
        goal = st.selectbox(
            "Primary Goal",
            ["weight_loss", "muscle_gain", "maintenance"]
        )
        preference = st.selectbox(
            "Diet Preference",
            ["none", "vegetarian"]   # "none" means no restriction
        )
        experience = st.slider(
            "Exercise Experience (1 = beginner, 4 = expert)",
            min_value=1, max_value=4, value=2, step=1
        )
        mode = st.selectbox(
            "Diet Mode",
            ["normal", "diabetic", "high_protein"]
        )
        conditions = st.multiselect(
            "Medical Conditions (if any)",
            ["diabetes", "heart", "cholesterol", "kidney"]
        )

        generate = st.button("Generate My Plan", type="primary")

    if generate:
        with st.spinner("Creating your personalized plan..."):
            # Convert preference: "none" -> None, otherwise keep as is
            pref = None if preference == "none" else preference
            # Conditions: if empty list, pass None
            conds = conditions if conditions else None

            result = health_fitness_system(
                age=age,
                gender=gender,
                weight=weight,
                height=height,
                activity_level=activity_level,
                goal=goal,
                preference=pref,
                experience=experience,
                mode=mode,
                conditions=conds,
                df=df,
                models=models
            )

        # Display results
        st.success("Plan generated successfully!")

        # BMI
        st.subheader(f"📊 BMI Category: **{result['BMI Class']}**")

        # Exercise Plan
        st.subheader("💪 Exercise Plan")
        ex = result["Exercise Plan"]
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Workout Type", ex["Workout Type"])
        with col2:
            st.metric("Frequency", f"{ex['Workout Frequency (days/week)']} days/week")
        with col3:
            st.metric("Session Duration", f"{ex['Session Duration (minutes)']} min")
        with col4:
            st.metric("Calories Burned", f"{ex['Estimated Calories Burned']} kcal")

        # Diet Plan
        st.subheader("🍽️ Daily Diet Plan")
        diet = result["Recommended Diet"]
        meals_order = ["breakfast", "lunch", "dinner", "snacks"]
        for meal in meals_order:
            df_meal = diet.get(meal)
            if df_meal is not None and not df_meal.empty:
                with st.expander(f"**{meal.title()}**", expanded=True):
                    # Show each food item (only food name, as requested)
                    for _, row in df_meal.iterrows():
                        st.markdown(f"• **{row['food']}**")
            else:
                with st.expander(f"**{meal.title()}**", expanded=False):
                    st.info("No foods selected for this meal.")

        # Optional: Show full daily totals
        all_meals_df = pd.concat([diet[m] for m in meals_order if not diet[m].empty], ignore_index=True)
        if not all_meals_df.empty:
            total_day_cal = all_meals_df["Caloric Value"].sum()
            total_prot = all_meals_df["Protein"].sum()
            total_carb = all_meals_df["Carbohydrates"].sum()
            total_fat = all_meals_df["Fat"].sum()
            st.subheader("📈 Daily Totals")
            st.markdown(f"**Total Calories:** {total_day_cal:.0f} kcal  \n"
                        f"**Protein:** {total_prot:.1f}g  \n"
                        f"**Carbohydrates:** {total_carb:.1f}g  \n"
                        f"**Fat:** {total_fat:.1f}g")
    else:
        st.info("👈 Fill in your details and click **Generate My Plan**")

if __name__ == "__main__":
    main()
