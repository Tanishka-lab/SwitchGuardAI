"""
=========================================================
SwitchGuardAI - Model Prediction
=========================================================

Loads the trained Random Forest model and makes predictions
on new sensor readings.

Two ways to use this:
1. --csv <path>  -> predicts for every row in a CSV file
2. --demo        -> runs a quick demo using two REAL rows
                    pulled from the test set (one Normal,
                    one a serious fault), just to show the
                    prediction pipeline working end-to-end

Author : Tanishka
Project : SwitchGuardAI
"""

import argparse
import os
import joblib
import pandas as pd

MODEL_PATH = "saved_models/random_forest.pkl"
FEATURE_NAMES_PATH = "saved_models/feature_names.pkl"

X_TEST_PATH = "data/processed/X_test.csv"
Y_TEST_PATH = "data/processed/y_test.csv"

RESULTS_FOLDER = "results"

NON_FEATURE_COLUMNS = [
    "Timestamp",
    "Switch_ID",
    "Health_Index",
    "Fault_Label"
]


def load_model_and_features():

    model = joblib.load(MODEL_PATH)
    feature_names = joblib.load(FEATURE_NAMES_PATH)

    return model, feature_names


def predict_from_csv(csv_path):
    """
    Predicts Fault_Label for every row in a CSV file.
    The CSV should already have the same engineered feature
    columns produced by feature_pipeline.py.
    """

    model, feature_names = load_model_and_features()

    new_data = pd.read_csv(csv_path)

    drop_cols = [c for c in NON_FEATURE_COLUMNS if c in new_data.columns]
    X_new = new_data.drop(columns=drop_cols)

    # make sure column order exactly matches what the model
    # was trained on - using the saved feature_names.pkl list
    X_new = X_new[feature_names]

    predictions = model.predict(X_new)
    probabilities = model.predict_proba(X_new)

    results = new_data.copy()
    results["Predicted_Fault"] = predictions

    for i, label in enumerate(model.classes_):
        results[f"Prob_{label}"] = probabilities[:, i]

    print(results[["Predicted_Fault"] + [f"Prob_{l}" for l in model.classes_]].head(10))

    os.makedirs(RESULTS_FOLDER, exist_ok=True)
    output_path = f"{RESULTS_FOLDER}/predictions.csv"
    results.to_csv(output_path, index=False)

    print(f"\nFull predictions saved to: {output_path}")

    return results


def predict_demo():
    """
    Demo using two REAL rows from the test set (one Normal,
    one a serious fault) - this avoids the problem of hand-typed
    fake rows having internally inconsistent feature values,
    which can confuse the model since it never saw such
    physically-inconsistent combinations during training.
    """

    model, feature_names = load_model_and_features()

    X_test = pd.read_csv(X_TEST_PATH)
    y_test = pd.read_csv(Y_TEST_PATH).squeeze()

    normal_idx = y_test[y_test == "Normal"].index[0]

    fault_candidates = y_test[y_test == "Critical"].index
    if len(fault_candidates) == 0:
        fault_candidates = y_test[y_test != "Normal"].index
    fault_idx = fault_candidates[0]

    demo_df = X_test.loc[[normal_idx, fault_idx], feature_names]
    true_labels = y_test.loc[[normal_idx, fault_idx]]

    predictions = model.predict(demo_df)
    probabilities = model.predict_proba(demo_df)

    print("Demo Prediction 1:")
    print("  Actual   :", true_labels.iloc[0])
    print("  Predicted:", predictions[0])

    print("\nDemo Prediction 2:")
    print("  Actual   :", true_labels.iloc[1])
    print("  Predicted:", predictions[1])

    print("\nFull probability breakdown:")
    prob_df = pd.DataFrame(probabilities, columns=model.classes_)
    print(prob_df.round(3))

    return predictions, probabilities


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="SwitchGuardAI prediction tool")
    parser.add_argument("--csv", type=str, help="Path to a CSV of new sensor readings")
    parser.add_argument("--demo", action="store_true", help="Run a quick demo prediction")

    args = parser.parse_args()

    if args.csv:
        predict_from_csv(args.csv)
    elif args.demo:
        predict_demo()
    else:
        print("Please provide --csv <path> or --demo")