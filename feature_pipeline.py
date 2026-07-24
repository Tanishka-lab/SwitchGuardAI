"""
=========================================================
SwitchGuardAI - Feature Engineering Pipeline (Main)
=========================================================

Runs all feature engineering stages in order:

1. feature_builder.py    -> physics-based features
2. rolling_features.py   -> moving averages
3. temporal_features.py  -> lag/gradient features
4. feature_selector.py   -> correlation analysis (reporting only)

Then encodes categorical columns and saves the final,
model-ready dataset.

Author : Tanishka
Project : SwitchGuardAI
"""

import os
import pandas as pd

from feature_engineering.feature_builder import build_physics_features
from feature_engineering.rolling_features import build_rolling_features
from feature_engineering.temporal_features import build_temporal_features
from feature_engineering.feature_selector import select_features

INPUT_PATH = "data/processed/switch_machine_dataset_clean.csv"
OUTPUT_PATH = "data/processed/switch_machine_dataset_features.csv"

ROLLING_WINDOW = 5


def run_pipeline():

    print("=" * 60)
    print("SwitchGuardAI Feature Engineering Pipeline")
    print("=" * 60)

    # -------------------------------------------------
    # LOAD
    # -------------------------------------------------

    df = pd.read_csv(INPUT_PATH)
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])

    # CRITICAL: sort per switch by time before any
    # rolling/lag/gradient feature is built
    df = df.sort_values(["Switch_ID", "Timestamp"]).reset_index(drop=True)

    print("Loaded shape:", df.shape)

    # -------------------------------------------------
    # STAGE 1: physics-based features
    # -------------------------------------------------

    print("\nStage 1: building physics-based features...")
    df = build_physics_features(df)

    # -------------------------------------------------
    # STAGE 2: rolling features
    # -------------------------------------------------

    print("Stage 2: building rolling features...")
    df = build_rolling_features(df, window=ROLLING_WINDOW)

    # -------------------------------------------------
    # STAGE 3: temporal / lag features
    # -------------------------------------------------

    print("Stage 3: building temporal/lag features...")
    df = build_temporal_features(df)

    # -------------------------------------------------
    # Time-based features (hour of day, is night)
    # -------------------------------------------------

    df["Hour_of_Day"] = df["Timestamp"].dt.hour

    df["Day_of_Week"] = df["Timestamp"].dt.dayofweek

    df["Month"] = df["Timestamp"].dt.month

    df["Is_Weekend"] = (
      df["Day_of_Week"] >= 5
    ).astype(int)

    df["Is_Night"] = (
      (df["Hour_of_Day"] < 6) |
      (df["Hour_of_Day"] >= 20)
    ).astype(int)

    # -------------------------------------------------
    # STAGE 4: encode categorical columns
    # (Season, Location_Type, and the new Previous_Fault
    # lag feature all need encoding for ML use)
    # -------------------------------------------------

    print("Stage 4: encoding categorical columns...")
    df = pd.get_dummies(
        df,
        columns=["Season", "Location_Type", "Previous_Fault"],
        prefix=["Season", "Location", "PrevFault"]
    )

    print("\nTotal columns after feature engineering:", len(df.columns))
    print("Missing values remaining:", df.isnull().sum().sum())

    # -------------------------------------------------
    # Final sort before saving
    # -------------------------------------------------

    df = df.sort_values(
     ["Switch_ID", "Timestamp"]
    ).reset_index(drop=True)

    # -------------------------------------------------
    # SAVE
    # -------------------------------------------------

    os.makedirs("data/processed", exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    print(f"\nSaved feature-engineered dataset to: {OUTPUT_PATH}")
    print("Final shape:", df.shape)

    # -------------------------------------------------
    # STAGE 5: feature selection / correlation report
    # (reporting only - does not change the saved file)
    # -------------------------------------------------

    print("\nStage 5: correlation analysis (Health_Index is used")
    print("here for ANALYSIS ONLY, never as a model input feature)\n")

    select_features(
        df,
        target_col="Health_Index",
        exclude=["Operation_Count"],
        top_n=15
    )

    return df


if __name__ == "__main__":
    run_pipeline()