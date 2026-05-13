import os
import joblib
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

# Configuration
DATA_PATH = "data/processed/pbp_2024_success_ml_data.parquet"
OUT_DIR = "artifacts"
os.makedirs(OUT_DIR, exist_ok=True)

# The columns the model "learns" from
NUM_COLS = ["week", "down", "ydstogo", "yardline_100", "score_differential",
            "game_seconds_remaining", "defenders_in_box", "number_of_pass_rushers"]
CAT_COLS = ["posteam", "defteam", "offense_formation_clean",
            "surface_clean", "roof_clean", "weather_ui", "score_state"]
FEATURES = NUM_COLS + CAT_COLS


def train_pre_snap_model(df, family_name):
    df = df.dropna(subset=["success"] + NUM_COLS).copy()

    for col in CAT_COLS:
        # Strip spaces and make everything uppercase
        df[col] = df[col].astype(str).str.strip().str.upper()
        # convert to category
        df[col] = df[col].replace("NAN", "UNKNOWN").astype("category")

    cat_mask = [col in CAT_COLS for col in FEATURES]
    model = HistGradientBoostingClassifier(categorical_features=cat_mask, max_depth=5)
    model.fit(df[FEATURES], df["success"].astype(int))

    joblib.dump(model, os.path.join(OUT_DIR, f"{family_name.lower()}_model.joblib"))

if __name__ == "__main__":
    full_df = pd.read_parquet(DATA_PATH)
    for fam in ["PASS_FAMILY", "RUSH"]:
        train_pre_snap_model(full_df[full_df["play_family"] == fam], fam)

# from pathlib import Path
# import joblib
# import pandas as pd
# from sklearn.ensemble import HistGradientBoostingClassifier
# from sklearn.metrics import roc_auc_score
#
# BASE_DIR = Path(__file__).parent
#
# DATA_PATH = r"C:\Users\might\OneDrive\Email attachments\pbp_2024.csv"
# MODEL_PATH = BASE_DIR / "success_model.pkl"
#
# # The columns the model "learns" from
# NUM_COLS = ["down", "ydstogo", "yardline_100", "score_differential", "temp", "wind",
#             "qtr", "quarter_seconds_remaining", "defenders_in_box", "number_of_pass_rushers"]
# CAT_COLS = ["posteam", "defteam", "offense_formation"]
# FEATURES = NUM_COLS + CAT_COLS
#
#
# def train_success_model():
#     # Drop rows where target or features are missing
#     df = pd.read_csv(DATA_PATH)
#
#     if "week" not in df.columns:
#         raise ValueError("Week column is missing")
#
#     df["week"] = df["week"].astype(int)
#     df = df.dropna(subset=["success"] + NUM_COLS + CAT_COLS).copy()
#
#     for col in CAT_COLS:
#         # Standardize categorical strings
#         df[col] = df[col].astype(str).str.strip().str.upper()
#         df[col] = df[col].replace("NAN", "UNKNOWN").astype("category")
#
#     train_df = df[df["week"] <= 14].copy()
#     test_df = df[df["week"] >= 15].copy()
#
#     if len(train_df) == 0 or len(test_df) == 0:
#         raise ValueError("Week split produced empty train or test data")
#
#     # Create the categorical mask for HistGradientBoosting
#     cat_mask = [col in CAT_COLS for col in FEATURES]
#
#     print(f"Training model on {len(df)} plays...")
#     model = HistGradientBoostingClassifier(categorical_features=cat_mask, max_depth=5)
#     model.fit(train_df[FEATURES], train_df["success"].astype(int))
#
#     proba = model.predict_proba(test_df[FEATURES])[:, 1]
#     y_true = test_df["success"].astype(int)
#     print(f"AUC: {roc_auc_score(y_true, proba)}")
#
#     # Save artifact as pkl
#     joblib.dump(model, MODEL_PATH, compress=3)
#
#
# if __name__ == "__main__":
#     train_success_model()
