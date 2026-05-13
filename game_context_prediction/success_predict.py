import joblib
import pandas as pd

# Load models
MODELS = {
    "PASS_FAMILY": joblib.load("artifacts/pass_family_model.joblib"),
    "RUSH": joblib.load("artifacts/rush_model.joblib")
}

def calculate_game_seconds(qtr, qtr_seconds):
    # Each quarter is 900 seconds
    quarters_left = 4 - qtr
    return (quarters_left * 900) + qtr_seconds

def get_prediction(user_input):
    # Convert user inputs to model features
    # Map the "Behind/Neutral/Ahead" to a score differential proxy if unknown
    score_diff_map = {"behind": -7, "neutral": 0, "ahead": 7}

    processed_input = {
        "week": 17,  # Defaulting to late season
        "down": user_input["downs"],
        "ydstogo": user_input["yards_to_go"],
        "yardline_100": user_input["field_position"],
        "score_differential": score_diff_map.get(user_input["score_state"].lower(), 0),
        "game_seconds_remaining": calculate_game_seconds(user_input["quarter"], user_input["seconds_left_in_quarter"]),
        "defenders_in_box": user_input["defenders_in_box"],
        "number_of_pass_rushers": user_input["number_of_pass_rushers"],
        "posteam": user_input["offense_team"],
        "defteam": user_input["defense_team"],
        "offense_formation_clean": user_input["offense_formation"],
        "surface_clean": user_input["field_type"],
        "roof_clean": user_input["outdoor_indoor"],
        "weather_ui": user_input["weather"],
        "score_state": user_input["score_state"]
    }

    # Convert to df and set categories
    input_df = pd.DataFrame([processed_input])
    cat_cols = ["posteam", "defteam", "offense_formation_clean", "surface_clean", "roof_clean", "weather_ui",
                "score_state"]
    for col in cat_cols:
        input_df[col] = input_df[col].astype("category")

    # Predict
    model = MODELS.get(user_input["play_type"])
    prob = model.predict_proba(input_df)[0, 1]
    return f"{prob * 100:.2f}%"


# EXAMPLE
user_params = {
    "offense_team": "KC",
    "defense_team": "SF",
    "play_type": "PASS_FAMILY",
    "weather": "mild",
    "field_type": "grass",
    "outdoor_indoor": "outdoor",
    "field_position": 20,
    "downs": 4,
    "yards_to_go": 15,
    "score_state": "neutral",
    "defenders_in_box": 7,
    "number_of_pass_rushers": 4,
    "quarter": 1,
    "seconds_left_in_quarter": 900,  # Start of game
    "offense_formation": "SHOTGUN"
}

print(f"Calculated Success Probability: {get_prediction(user_params)}")
#
# from pathlib import Path
# import joblib
# import pandas as pd
# import os
# from success_train import train_success_model
#
# BASE_DIR = Path(__file__).parent
#
# MODEL_PATH = BASE_DIR / "success_model.pkl"
#
#
# def ensure_model_exists():
#     if not os.path.exists(MODEL_PATH):
#         print("Model not found. Training model locally...")
#         train_success_model()
#         print("Training complete.")
#
#
# # Ensure model before loading
# ensure_model_exists()
#
# # Load the single success model
# MODEL = joblib.load(MODEL_PATH)
#
#
# def prediction_success(
#         down, ydstogo, yardline_100, score_differential,
#         temp, wind, qtr, qtr_seconds, defenders_in_box,
#         number_of_pass_rushers, posteam, defteam,
#         offense_formation
# ):
#     processed_input = {
#         "down": down,
#         "ydstogo": ydstogo,
#         "yardline_100": yardline_100,
#         "score_differential": score_differential,
#         "temp": temp,
#         "wind": wind,
#         "qtr": qtr,
#         "quarter_seconds_remaining": qtr_seconds,
#         "defenders_in_box": defenders_in_box,
#         "number_of_pass_rushers": number_of_pass_rushers,
#         "posteam": str(posteam).strip().upper(),
#         "defteam": str(defteam).strip().upper(),
#         "offense_formation": str(offense_formation).strip().upper()
#     }
#
#     # Convert to DataFrame
#     input_df = pd.DataFrame([processed_input])
#
#     # Cast categorical columns to match training set expectations
#     cat_cols = ["posteam", "defteam", "offense_formation"]
#     for col in cat_cols:
#         input_df[col] = input_df[col].astype("category")
#
#     # Predict
#     prob = MODEL.predict_proba(input_df)[0, 1]
#     return f"{prob * 100:.2f}%"
