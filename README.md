# AI Meal Planner App

This project is a meal planning app that generates personalized meal plans based on a user's caloric needs and food
preferences.
It uses Llama-3 70B through Groq API to generate creative meal ideas using ingredients selected by the algorithm.

### Try the app [here](https://ai-meal-planner.streamlit.app)

## Demo

![image](https://github.com/myselfshravan/AI-Meal-Planner/assets/71520844/b33ba708-ded4-467d-8e0a-68f5e47e4b80)
![image](https://github.com/myselfshravan/AI-Meal-Planner/assets/71520844/5d1d86ed-d7f5-46a0-9dfe-42b2d43bae90)
![image](https://github.com/myselfshravan/AI-Meal-Planner/assets/71520844/895511ea-7e07-4c6c-827b-2fa948f47623)

## Features

- Calculates daily calorie needs based on user inputs (age, height, weight, gender)
- Supports both metric (kg, cm) and imperial (lb, ft + in) units
- Allows selection of food preferences and allergies/restrictions
- Generates personalized meal plans (breakfast, lunch, dinner) using knapsack algorithm to optimize calorie targets
- Creates creative meal names and descriptions using Llama-3 70B AI model
- Displays detailed nutritional information for each meal

## Technology

- Python
- Streamlit for web interface
- Pandas for data manipulation
- Llama-3 70B via Groq API for AI text generation
- Knapsack algorithm for optimal meal planning

Add your API key to `.streamlit/secrets.toml`:

```bash
GROQ_API_KEY="YOUR_API_KEY"
```

## Meal Planning Algorithm

The app uses the Knapsack algorithm to optimize meal planning:

1. Calculates target calories for breakfast (50%), lunch (33%), and dinner (17%) based on BMR
2. For each meal:
   - Considers all available food items as potential choices
   - Optimizes selection to maximize calories while staying under target
   - Uses dynamic programming to find the optimal combination of items
   - Returns selected food items and total calories

### The Knapsack algorithm ensures optimal food selection to meet calorie targets while considering all possible combinations.

---

The app calculates the user's basal metabolic rate (BMR) to determine daily calorie needs. It then uses the Knapsack algorithm to select optimal food combinations for each meal. The selected ingredients are passed to the Llama-3 70B model via Groq API to generate creative meal names and descriptions.

The project demonstrates practical application of AI and optimization algorithms for personalized meal planning. Future enhancements could include user accounts, expanded food database, detailed recipes, and meal prep instructions.
