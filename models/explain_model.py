"""
=========================================================
SwitchGuardAI - Explainable AI
=========================================================

Generates explainability reports for the trained
Random Forest model.

Outputs:
---------
1. Feature Importance
2. SHAP Summary Plot
3. SHAP Waterfall Plot (Single Prediction)
4. SHAP Values CSV

Author : Tanishka
Project : SwitchGuardAI
"""

import os
import joblib
import shap
import pandas as pd
import matplotlib.pyplot as plt

MODEL_PATH = "saved_models/random_forest.pkl"
X_TEST_PATH = "data/processed/X_test.csv"

RESULTS_DIR = "results"


def explain_model():

    print("=" * 60)
    print("SwitchGuardAI - Explainable AI")
    print("=" * 60)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("Loading model...")

    model = joblib.load(MODEL_PATH)

    X_test = pd.read_csv(X_TEST_PATH)

    print("Test Samples :", len(X_test))

    # =====================================================
    # FEATURE IMPORTANCE
    # =====================================================

    print("\nGenerating Feature Importance...")

    importance = pd.Series(
        model.feature_importances_,
        index=X_test.columns
    ).sort_values(ascending=False)

    importance.to_csv(
        os.path.join(
            RESULTS_DIR,
            "feature_importance.csv"
        )
    )

    plt.figure(figsize=(10, 8))

    importance.head(15).sort_values().plot.barh()

    plt.title("Top 15 Feature Importance")

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            RESULTS_DIR,
            "feature_importance.png"
        ),
        dpi=300
    )

    plt.close()

    # =====================================================
    # SHAP
    # =====================================================

    print("\nGenerating SHAP values...")

    explainer = shap.TreeExplainer(model)

    shap_values = explainer(X_test)

    # =====================================================
    # SHAP SUMMARY PLOT
    # =====================================================

    print("Generating SHAP Summary Plot...")

    # Explain the Critical class globally
    critical_index = list(model.classes_).index("Critical")

    shap.summary_plot(
        shap_values[:, :, critical_index],
        X_test,
        show=False
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            RESULTS_DIR,
            "shap_summary.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # =====================================================
    # SHAP WATERFALL (Single Prediction)
    # =====================================================

    print("Generating SHAP Waterfall Plot...")

    sample_index = 0

    prediction = model.predict(
        X_test.iloc[[sample_index]]
    )[0]

    predicted_class_index = list(model.classes_).index(prediction)

    explanation = shap_values[
        sample_index,
        :,
        predicted_class_index
    ]

    plt.figure(figsize=(10, 6))

    shap.plots.waterfall(
        explanation,
        show=False
    )

    plt.savefig(
        os.path.join(
            RESULTS_DIR,
            "shap_sample_prediction.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # =====================================================
    # SAVE SHAP VALUES
    # =====================================================

    print("Saving SHAP values...")

    shap_df = pd.DataFrame(
        shap_values.values[:, :, critical_index],
        columns=X_test.columns
    )

    shap_df.to_csv(
        os.path.join(
            RESULTS_DIR,
            "shap_values.csv"
        ),
        index=False
    )

    # =====================================================

    print("\nExplainability Complete.")

    print("\nFiles Generated")
    print("----------------------------")
    print("feature_importance.csv")
    print("feature_importance.png")
    print("shap_summary.png")
    print("shap_sample_prediction.png")
    print("shap_values.csv")
    print("----------------------------")


if __name__ == "__main__":
    explain_model()