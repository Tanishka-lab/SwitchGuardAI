"""
=========================================================
SwitchGuardAI - models/train_model.py
=========================================================

Trains a Random Forest classifier to predict Fault_Label
from the engineered sensor features, and saves the model
+ test data for the evaluation/explainability steps.

Author : Tanishka
Project : SwitchGuardAI
"""

import os
import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier

from models.data_split import (
    split_train_test,
    get_features_and_target
)

DATA_PATH = "data/processed/switch_machine_dataset_features.csv"
MODEL_OUTPUT_PATH = "saved_models/random_forest.pkl"


def train_model():

    print("=" * 60)
    print("SwitchGuardAI - Model Training (Random Forest)")
    print("=" * 60)

    # -------------------------------------------------
    # LOAD + SPLIT
    # -------------------------------------------------

    df = pd.read_csv(DATA_PATH)

    train_df, test_df = split_train_test(df, test_fraction=0.2)

    X_train, y_train = get_features_and_target(train_df)
    X_test, y_test = get_features_and_target(test_df)

    print("Train shape:", X_train.shape)
    print("Test shape :", X_test.shape)

    # -------------------------------------------------
    # TRAIN
    # -------------------------------------------------

    model = RandomForestClassifier(
        n_estimators=200,          # number of decision trees to build
        max_depth=None,            # let trees grow fully
        class_weight="balanced",   # pay more attention to rare fault classes
        random_state=42,           # reproducible results
        n_jobs=-1                  # use all CPU cores to train faster
    )

    print("\nTraining Random Forest...")
    model.fit(X_train, y_train)
    print("Training complete.")

    # -------------------------------------------------
    # SAVE MODEL + TEST DATA (used by evaluate_model.py
    # and explain_model.py)
    # -------------------------------------------------

    os.makedirs("saved_models", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)

    joblib.dump(model, MODEL_OUTPUT_PATH)

    joblib.dump(
    list(X_train.columns),
    "saved_models/feature_names.pkl"
    )

    X_train.to_csv("data/processed/X_train.csv", index=False)
    y_train.to_csv("data/processed/y_train.csv", index=False)
    X_test.to_csv("data/processed/X_test.csv", index=False)
    y_test.to_csv("data/processed/y_test.csv", index=False)

    print(f"\nModel saved to: {MODEL_OUTPUT_PATH}")
    print("Train/test data saved for the next steps.")

    return model, X_train, y_train, X_test, y_test


if __name__ == "__main__":
    train_model()