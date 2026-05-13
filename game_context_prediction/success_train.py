import os
import joblib
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Configuration
DATA_PATH = "data/processed/pbp_2024_success_ml_data.parquet"
OUT_DIR = "artifacts"
os.makedirs(OUT_DIR, exist_ok=True)

NUM_COLS = ["down", "ydstogo", "yardline_100", "score_differential",
            "game_seconds_remaining", "defenders_in_box", "number_of_pass_rushers"]
CAT_COLS = ["posteam", "defteam", "offense_formation_clean",
            "surface_clean", "roof_clean", "weather_ui", "score_state"]
FEATURES = NUM_COLS + CAT_COLS

def train_pre_snap_model(df, family_name):
    # Clean Data
    df = df.dropna(subset=["success"] + NUM_COLS).copy()
    for col in CAT_COLS:
        df[col] = df[col].astype(str).str.strip().str.upper()
        df[col] = df[col].replace("NAN", "UNKNOWN").astype("category")

    # Train/Test Split
    X = df[FEATURES]
    y = df["success"].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Training
    cat_mask = [col in CAT_COLS for col in FEATURES]
    model = HistGradientBoostingClassifier(categorical_features=cat_mask, max_depth=5)
    model.fit(X_train, y_train)

    # Evaluation
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n--- {family_name} Model Results ---")
    print(f"Accuracy: {acc:.2%}")
    print(classification_report(y_test, y_pred))

    # Save Model
    joblib.dump(model, os.path.join(OUT_DIR, f"{family_name.lower()}_model.joblib"))

if __name__ == "__main__":
    full_df = pd.read_parquet(DATA_PATH)
    for fam in ["PASS_FAMILY", "RUSH"]:
        train_pre_snap_model(full_df[full_df["play_family"] == fam], fam)
