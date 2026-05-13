import os
import joblib
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

# Configuration
DATA_PATH = "data/processed/pbp_2024_success_ml_data.parquet"
OUT_DIR = "artifacts"
os.makedirs(OUT_DIR, exist_ok=True)

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
