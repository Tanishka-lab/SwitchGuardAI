"""
=========================================================
SwitchGuardAI - Explainability Dashboard
=========================================================
"""

import streamlit as st
import joblib
import shap
import pandas as pd
import matplotlib.pyplot as plt

MODEL_PATH = "saved_models/random_forest.pkl"
FEATURE_PATH = "saved_models/feature_names.pkl"
X_TEST_PATH = "data/processed/X_test.csv"


# ==========================================================
# LOAD RESOURCES
# ==========================================================

@st.cache_resource
def load_resources():

    model = joblib.load(MODEL_PATH)

    feature_names = joblib.load(FEATURE_PATH)

    X_test = pd.read_csv(X_TEST_PATH)

    explainer = shap.TreeExplainer(model)

    return model, feature_names, X_test, explainer


# ==========================================================
# PAGE
# ==========================================================

def show_explainability():

    st.title("🔍 Model Explainability")

    st.write(
        """
Understand **why** the AI predicted a particular fault.

Instead of acting as a black box, SwitchGuardAI highlights
the sensor readings responsible for every prediction using
**SHAP (SHapley Additive Explanations)**.
"""
    )

    model, feature_names, X_test, explainer = load_resources()

    sample = st.slider(
        "Select Test Sample",
        0,
        len(X_test) - 1,
        0
    )

    instance = X_test.iloc[[sample]]

    prediction = model.predict(instance)[0]

    probabilities = model.predict_proba(instance)[0]

    confidence = probabilities.max() * 100

    st.divider()

    c1, c2 = st.columns(2)

    with c1:
        st.metric(
            "Predicted Fault",
            prediction
        )

    with c2:
        st.metric(
            "Confidence",
            f"{confidence:.2f}%"
        )

    st.divider()

    # =====================================================
    # SHAP VALUES
    # =====================================================

    shap_values = explainer(instance)

    class_index = list(model.classes_).index(prediction)

    explanation = shap_values[:, :, class_index]

    values = explanation.values[0]

    base_value = explanation.base_values[0]

    # =====================================================
    # TOP FEATURES
    # =====================================================

    importance = pd.DataFrame({

        "Feature": feature_names,

        "SHAP Value": values,

        "Contribution": abs(values)

    })

    importance = importance.sort_values(
        "Contribution",
        ascending=False
    )

    top10 = importance.head(10)

    st.subheader("Top Feature Contributions")

    st.bar_chart(
        top10.set_index("Feature")["Contribution"]
    )

    # =====================================================
    # POSITIVE / NEGATIVE
    # =====================================================

    positive = importance[
        importance["SHAP Value"] > 0
    ].head(5)

    negative = importance[
        importance["SHAP Value"] < 0
    ].sort_values(
        "SHAP Value"
    ).head(5)

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("🔴 Increased Fault Probability")

        if len(positive) == 0:

            st.write("No positive contributors.")

        else:

            for _, row in positive.iterrows():

                st.write(
                    f"• **{row.Feature}** (+{row['SHAP Value']:.3f})"
                )

    with col2:

        st.subheader("🟢 Reduced Fault Probability")

        if len(negative) == 0:

            st.write("No negative contributors.")

        else:

            for _, row in negative.iterrows():

                st.write(
                    f"• **{row.Feature}** ({row['SHAP Value']:.3f})"
                )

    st.divider()

    # =====================================================
    # WATERFALL
    # =====================================================

    st.subheader("SHAP Waterfall Plot")

    waterfall_exp = shap.Explanation(

        values=values,

        base_values=base_value,

        data=instance.iloc[0].values,

        feature_names=feature_names

    )

    fig = plt.figure(figsize=(10, 6))

    shap.plots.waterfall(
        waterfall_exp,
        show=False
    )

    st.pyplot(fig)

    plt.close()

    st.divider()

    # =====================================================
    # INTERPRETATION
    # =====================================================

    top_feature = top10.iloc[0]["Feature"]

    st.subheader("Interpretation")

    st.info(
        f"""
**Prediction:** {prediction}

The model was influenced most by **{top_feature}**.

The chart above shows the ten most important engineered
features for this prediction.

Positive SHAP values push the prediction toward
**{prediction}**, while negative SHAP values push it away.

This allows maintenance engineers to understand **why**
SwitchGuardAI produced its prediction instead of treating
the model as a black box.
"""
    )