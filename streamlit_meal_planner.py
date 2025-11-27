import streamlit as st
import pandas as pd
import random
import time
import os
import json
from openai import OpenAI
from data import food_items_breakfast, food_items_lunch, food_items_dinner
from prompts import pre_prompt_b, pre_prompt_l, pre_prompt_d, pre_breakfast, pre_lunch, pre_dinner, end_text, \
    example_response_l, example_response_d, negative_prompt

UNITS_CM_TO_IN = 0.393701
UNITS_IN_TO_CM = 1 / UNITS_CM_TO_IN
UNITS_KG_TO_LB = 2.20462
UNITS_LB_TO_KG = 1 / UNITS_KG_TO_LB

# Configure Groq API using OpenAI client
client = OpenAI(
    api_key=st.secrets['GROQ_API_KEY'],
    base_url="https://api.groq.com/openai/v1"
)

# Disable SSL verification warnings
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


def _save_calendar_to_file(path="meal_calendar.json"):
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(st.session_state.get("meal_calendar", {}), f, ensure_ascii=False, indent=2)
        return True
    except Exception:
        return False


def _load_calendar_from_file(path="meal_calendar.json"):
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
    return None


def _get_item_calories(item_name: str) -> int:
    """Look up calories for an item across breakfast, lunch, and dinner catalogs.
    Handles comma-separated items (e.g., "eggs, oatmeal") by summing their calories."""
    if not item_name or item_name == "-":
        return 0
    
    total_cal = 0
    # Split by comma to handle multiple items
    items = [i.strip() for i in item_name.split(",")]
    for item in items:
        # skip "+X more" labels
        if "+" in item or "more" in item:
            continue
        # search breakfast
        for group in food_items_breakfast.values():
            if item in group:
                total_cal += int(group[item])
                break
        else:
            # search lunch
            for group in food_items_lunch.values():
                if item in group:
                    total_cal += int(group[item])
                    break
            else:
                # search dinner
                for group in food_items_dinner.values():
                    if item in group:
                        total_cal += int(group[item])
                        break
    return total_cal

st.set_page_config(page_title="AI - Meal Planner", page_icon="🍴")

st.title("AI Meal Planner")
st.divider()

st.write(
    "This is a AI based meal planner that uses a persons information. The planner can be used to find a meal plan that satisfies the user's calorie and macronutrient requirements.")
st.markdown("*Powered by Llama-3 70B*")

st.divider()

st.write("Enter your information:")
name = st.text_input("Enter your name")
age = st.number_input("Enter your age", step=1)

unit_preference = st.radio("Preferred units:", ["Metric (kg, cm)", "Imperial (lb, ft + in)"])

if unit_preference == "Metric (kg, cm)":
    weight = st.number_input("Enter your weight (kg)")
    height = st.number_input("Enter your height (cm)")
else:
    weight_lb = st.number_input("Enter your weight (lb)")
    
    # Use columns to align feet and inches inputs next to each other
    col1, col2 = st.columns(2)
    with col1:
        height_ft = st.number_input("Enter your height (ft)")
    with col2:
        height_in = st.number_input("Enter your height (in)")

    # Convert imperial to metric
    weight = weight_lb * UNITS_LB_TO_KG
    height = (height_ft * 12 + height_in) * UNITS_IN_TO_CM

gender = st.radio("Choose your gender:", ["Male", "Female"])
example_response = f"This is just an example but use your creativity: You can start with, Hello {name}! I'm thrilled to be your meal planner for the day, and I've crafted a delightful and flavorful meal plan just for you. But fear not, this isn't your ordinary, run-of-the-mill meal plan. It's a culinary adventure designed to keep your taste buds excited while considering the calories you can intake. So, get ready!"


def calculate_bmr(weight, height, age, gender):
    if gender == "Male":
        bmr = 9.99 * weight + 6.25 * height - 4.92 * age + 5
    else:
        bmr = 9.99 * weight + 6.25 * height - 4.92 * age - 161

    return bmr


def get_user_preferences():
    preferences = st.multiselect("Choose your food preferences:", list(food_items_breakfast.keys()))
    return preferences


def get_user_allergies():
    allergies = st.multiselect("Choose your food allergies:", list(food_items_breakfast.keys()))
    return allergies


def generate_items_list(target_calories, food_groups):
    calories = 0
    selected_items = []
    total_items = set()
    for foods in food_groups.values():
        total_items.update(foods.keys())

    while abs(calories - target_calories) >= 10 and len(selected_items) < len(total_items):
        group = random.choice(list(food_groups.keys()))
        foods = food_groups[group]
        item = random.choice(list(foods.keys()))

        if item not in selected_items:
            cals = foods[item]
            if calories + cals <= target_calories:
                selected_items.append(item)
                calories += cals

    return selected_items, calories


def knapsack(target_calories, food_groups):
    items = []
    for group, foods in food_groups.items():
        for item, calories in foods.items():
            items.append((calories, item))

    n = len(items)
    dp = [[0 for _ in range(target_calories + 1)] for _ in range(n + 1)]

    for i in range(1, n + 1):
        for j in range(target_calories + 1):
            value, _ = items[i - 1]

            if value > j:
                dp[i][j] = dp[i - 1][j]
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i - 1][j - value] + value)

    selected_items = []
    j = target_calories
    for i in range(n, 0, -1):
        if dp[i][j] != dp[i - 1][j]:
            _, item = items[i - 1]
            selected_items.append(item)
            j -= items[i - 1][0]

    return selected_items, dp[n][target_calories]


bmr = calculate_bmr(weight, height, age, gender)
round_bmr = round(bmr, 2)
st.subheader(f"Your daily intake needs to have: {round_bmr} calories")
choose_algo = "Knapsack"
if 'clicked' not in st.session_state:
    st.session_state.clicked = False


def click_button():
    st.session_state.clicked = True


if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "llama-3.3-70b-versatile"
    # st.session_state["openai_model"] = "gpt-4o"

if "messages" not in st.session_state:
    st.session_state.messages = []

# Load previously saved calendar if available (persist between restarts)
if "meal_calendar" not in st.session_state:
    loaded = _load_calendar_from_file()
    if loaded:
        st.session_state["meal_calendar"] = loaded
        st.session_state["meal_calendar_gen"] = 0
    else:
        st.session_state["meal_calendar"] = {}

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

st.button("Create a Basket", on_click=click_button)
if st.session_state.clicked:
    calories_breakfast = round((bmr * 0.5), 2)
    calories_lunch = round((bmr * (1 / 3)), 2)
    calories_dinner = round((bmr * (1 / 6)), 2)

    if choose_algo == "Random Greedy":
        meal_items_morning, cal_m = generate_items_list(calories_breakfast, food_items_breakfast)
        meal_items_lunch, cal_l = generate_items_list(calories_lunch, food_items_lunch)
        meal_items_dinner, cal_d = generate_items_list(calories_dinner, food_items_dinner)

    else:
        meal_items_morning, cal_m = knapsack(int(calories_breakfast), food_items_breakfast)
        meal_items_lunch, cal_l = knapsack(int(calories_lunch), food_items_lunch)
        meal_items_dinner, cal_d = knapsack(int(calories_dinner), food_items_dinner)
    st.header("Your Personalized Meal Plan")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.write("Calories for Morning: " + str(calories_breakfast))
        st.dataframe(pd.DataFrame({"Morning": meal_items_morning}))
        st.write("Total Calories: " + str(cal_m))

    with col2:
        st.write("Calories for Lunch: " + str(calories_lunch))
        st.dataframe(pd.DataFrame({"Lunch": meal_items_lunch}))
        st.write("Total Calories: " + str(cal_l))

    with col3:
        st.write("Calories for Dinner: " + str(calories_dinner))
        st.dataframe(pd.DataFrame({"Dinner": meal_items_dinner}))
        st.write("Total Calories: " + str(cal_d))

    if st.button("Generate Meal Plan"):
        st.markdown("""---""")
        st.subheader("Breakfast")
        user_content = pre_prompt_b + str(meal_items_morning) + example_response + pre_breakfast + negative_prompt
        temp_messages = [{"role": "user", "content": user_content}]
        with st.chat_message("assistant"):
            full_response = ""
            try:
                response = client.chat.completions.create(
                    model=st.session_state["openai_model"],
                    messages=temp_messages
                )
                full_response = response.choices[0].message.content
                st.write(full_response)
            except Exception as e:
                error_msg = f"Error: {str(e)}"
                st.error(error_msg)
                full_response = error_msg
        st.session_state.messages.append(
            {"role": "assistant", "content": full_response})

        st.markdown("""---""")
        st.subheader("Lunch")
        user_content = pre_prompt_l + str(meal_items_lunch) + example_response + pre_lunch + negative_prompt
        temp_messages = [{"role": "user", "content": user_content}]
        with st.chat_message("assistant"):
            full_response = ""
            try:
                response = client.chat.completions.create(
                    model=st.session_state["openai_model"],
                    messages=temp_messages
                )
                full_response = response.choices[0].message.content
                st.write(full_response)
            except Exception as e:
                error_msg = f"Error: {str(e)}"
                st.error(error_msg)
                full_response = error_msg
        st.session_state.messages.append(
            {"role": "assistant", "content": full_response})

        st.markdown("""---""")
        st.subheader("Dinner")
        user_content = pre_prompt_d + str(meal_items_dinner) + example_response + pre_dinner + negative_prompt
        temp_messages = [{"role": "user", "content": user_content}]
        with st.chat_message("assistant"):
            full_response = ""
            try:
                response = client.chat.completions.create(
                    model=st.session_state["openai_model"],
                    messages=temp_messages
                )
                full_response = response.choices[0].message.content
                st.write(full_response)
            except Exception as e:
                error_msg = f"Error: {str(e)}"
                st.error(error_msg)
                full_response = error_msg
        st.session_state.messages.append(
            {"role": "assistant", "content": full_response})
        st.write("Thank you for using our AI app! I hope you enjoyed it!")

        # --- Meal Planner Calendar ---
        st.markdown("""---""")
        st.header("Meal Planner Calendar 🗓️")
        st.write("Assign your generated meals to each day of the week:")

        week_days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        default_meals = {
            "Breakfast": meal_items_morning or [],
            "Lunch": meal_items_lunch or [],
            "Dinner": meal_items_dinner or []
        }

        # Use a generation id so widget keys change every time a new meal plan is generated.
        gen_id = st.session_state.get("meal_generation_id", 0) + 1
        st.session_state["meal_generation_id"] = gen_id

        # Initialize calendar for this generation only if it hasn't been initialized
        # (prevents overwriting user selections when widgets cause reruns)
        if st.session_state.get("meal_calendar_gen") != gen_id:
            st.session_state["meal_calendar"] = {day: {meal: "-" for meal in default_meals} for day in week_days}
            st.session_state["meal_calendar_gen"] = gen_id

        # Auto-fill calendar with generated items (rotate lists across days) if available
        if st.session_state.get("meal_calendar_auto_filled_gen") != gen_id:
            any_items = any(len(v) > 0 for v in default_meals.values())
            if any_items:
                for idx, day in enumerate(week_days):
                    for meal_name, items in default_meals.items():
                        if items:
                            # Assign all items for this meal as a comma-separated list for the day
                            choice = ", ".join(items) if len(items) <= 3 else ", ".join(items[:3]) + f" (+{len(items)-3} more)"
                        else:
                            choice = "-"
                        st.session_state["meal_calendar"][day][meal_name] = choice
                st.session_state["meal_calendar_auto_filled_gen"] = gen_id

        # Quick assign controls: choose a meal option and apply to all days
        st.subheader("Quick Assign")
        apply_cols = st.columns(len(default_meals))
        for i, (meal_name, meal_opts) in enumerate(default_meals.items()):
            with apply_cols[i]:
                apply_options = ["-"] + list(meal_opts) if meal_opts else ["-"]
                apply_key = f"apply_all_{gen_id}_{meal_name}"
                apply_choice = st.selectbox(f"Set all {meal_name}", apply_options, key=apply_key)
                if st.button(f"Apply to all {meal_name}", key=f"apply_btn_{gen_id}_{meal_name}"):
                    for day in week_days:
                        widget_key = f"meal_{gen_id}_{day}_{meal_name}"
                        st.session_state[widget_key] = apply_choice
                        st.session_state["meal_calendar"][day][meal_name] = apply_choice

        for day in week_days:
            st.subheader(day)
            for meal in default_meals:
                options = ["-"] + list(default_meals[meal]) if default_meals[meal] else ["-"]
                # Use a unique key per generation so Streamlit recreates the widget when new items appear
                widget_key = f"meal_{gen_id}_{day}_{meal}"
                # Get current value from calendar (set by auto-fill or user selection)
                current_value = st.session_state["meal_calendar"][day][meal]
                # Find index if current value is in options, otherwise default to 0
                try:
                    default_index = options.index(current_value) if current_value in options else 0
                except ValueError:
                    default_index = 0
                selection = st.selectbox(f"{meal} for {day}", options, index=default_index, key=widget_key)
                st.session_state["meal_calendar"][day][meal] = selection

        # Display the calendar as a table and compute per-day calories
        st.markdown("**Your Weekly Meal Calendar:**")
        calendar_df = pd.DataFrame.from_dict(st.session_state["meal_calendar"], orient="index")

        # Compute calories per meal and per day
        breakfast_cals = []
        lunch_cals = []
        dinner_cals = []
        total_cals = []
        for day in calendar_df.index:
            b_item = calendar_df.loc[day].get("Breakfast", "-")
            l_item = calendar_df.loc[day].get("Lunch", "-")
            d_item = calendar_df.loc[day].get("Dinner", "-")

            b_cal = _get_item_calories(b_item)
            l_cal = _get_item_calories(l_item)
            d_cal = _get_item_calories(d_item)

            breakfast_cals.append(b_cal)
            lunch_cals.append(l_cal)
            dinner_cals.append(d_cal)
            total_cals.append(b_cal + l_cal + d_cal)

        calendar_df["Breakfast_Cal"] = breakfast_cals
        calendar_df["Lunch_Cal"] = lunch_cals
        calendar_df["Dinner_Cal"] = dinner_cals
        calendar_df["Total_Cal"] = total_cals

        st.dataframe(calendar_df)

        # Action buttons: Save, Export CSV, Clear
        col_save, col_export, col_clear = st.columns(3)
        with col_save:
            if st.button("Save calendar"):
                ok = _save_calendar_to_file()
                if ok:
                    st.success("Calendar saved to meal_calendar.json")
                else:
                    st.error("Failed to save calendar")
        with col_export:
            csv_data = calendar_df.to_csv(index=True)
            st.download_button("Download CSV", data=csv_data, file_name="meal_calendar.csv", mime="text/csv")
        with col_clear:
            if st.button("Clear calendar"):
                # clear selections for current generation
                for day in week_days:
                    for meal in default_meals:
                        widget_key = f"meal_{gen_id}_{day}_{meal}"
                        st.session_state[widget_key] = "-"
                        st.session_state["meal_calendar"][day][meal] = "-"
                st.experimental_rerun()

hide_streamlit_style = """
                    <style>
                    # MainMenu {visibility: hidden;}
                    footer {visibility: hidden;}
                    footer:after {
                    content:'Built by moin'; 
                    visibility: visible;
    	            display: block;
    	            position: relative;
    	            # background-color: red;
    	            padding: 15px;
    	            top: 2px;
    	            }
    	            #ai-meal-planner {
    	              text-align: center; !important
        	            }
                    </style>
                    """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)
