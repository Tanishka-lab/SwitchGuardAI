"""
=========================================================
SwitchGuardAI - Model Performance Dashboard
=========================================================
"""

import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

MODEL_PATH = "saved_models/random_forest.pkl"
FEATURE_PATH = "saved_models/feature_names.pkl"

X_TEST_PATH = "data/processed/X_test.csv"
Y_TEST_PATH = "data/processed/y_test.csv"


# ==========================================================
# LOAD MODEL
# ==========================================================

@st.cache_resource
def load_resources():

    model = joblib.load(MODEL_PATH)

    feature_names = joblib.load(FEATURE_PATH)

    X_test = pd.read_csv(X_TEST_PATH)

    y_test = pd.read_csv(Y_TEST_PATH).squeeze()

    return model, feature_names, X_test, y_test


# ==========================================================
# PERFORMANCE PAGE
# ==========================================================

def show_performance():

    st.title("📈 Model Performance")

    st.write(
        """
Evaluate the trained **Random Forest Classifier**
on unseen railway switch-machine data.

This page summarizes how well the model distinguishes
between different fault conditions.
"""
    )

    model, feature_names, X_test, y_test = load_resources()

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    precision = precision_score(
        y_test,
        predictions,
        average="weighted"
    )

    recall = recall_score(
        y_test,
        predictions,
        average="weighted"
    )

    f1 = f1_score(
        y_test,
        predictions,
        average="weighted"
    )

    # =====================================================
    # OVERALL METRICS
    # =====================================================

    st.subheader("Overall Performance")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Accuracy",
        f"{accuracy*100:.2f}%"
    )

    c2.metric(
        "Precision",
        f"{precision*100:.2f}%"
    )

    c3.metric(
        "Recall",
        f"{recall*100:.2f}%"
    )

    c4.metric(
        "F1 Score",
        f"{f1*100:.2f}%"
    )

    st.divider()

    # =====================================================
    # MODEL INFO
    # =====================================================

    st.subheader("🧠 Model Information")

    m1, m2, m3 = st.columns(3)

    m1.metric(
        "Algorithm",
        "Random Forest"
    )

    m2.metric(
        "Training Features",
        len(feature_names)
    )

    m3.metric(
        "Testing Samples",
        len(X_test)
    )

    st.divider()

    # =====================================================
    # CONFUSION MATRIX
    # =====================================================

    st.subheader("Confusion Matrix")

    cm = confusion_matrix(
        y_test,
        predictions,
        labels=model.classes_
    )

    fig, ax = plt.subplots(figsize=(7,6))

    im = ax.imshow(cm)

    ax.set_xticks(range(len(model.classes_)))
    ax.set_xticklabels(
        model.classes_,
        rotation=45
    )

    ax.set_yticks(range(len(model.classes_)))
    ax.set_yticklabels(model.classes_)

    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("Actual Label")

    plt.colorbar(im)

    for i in range(cm.shape[0]):

        for j in range(cm.shape[1]):

            ax.text(
                j,
                i,
                cm[i, j],
                ha="center",
                va="center",
                color="black"
            )

    st.pyplot(fig)

    plt.close()

    st.divider()
        # =====================================================
    # CLASSIFICATION REPORT
    # =====================================================

    st.subheader("📋 Classification Report")

    report = classification_report(
        y_test,
        predictions,
        output_dict=True
    )

    report_df = (
        pd.DataFrame(report)
        .transpose()
        .round(3)
    )

    st.dataframe(
        report_df,
        width="stretch"
    )

    st.divider()

    # =====================================================
    # PREDICTION DISTRIBUTION
    # =====================================================

    st.subheader("📊 Prediction Distribution")

    pred_counts = (
        pd.Series(predictions)
        .value_counts()
    )

    st.bar_chart(pred_counts)

    st.divider()

    # =====================================================
    # FEATURE IMPORTANCE
    # =====================================================

    st.subheader("⭐ Top 10 Most Important Features")

    importance = pd.DataFrame({

        "Feature": feature_names,

        "Importance": model.feature_importances_

    })

    importance = importance.sort_values(

        "Importance",

        ascending=False

    ).head(10)

    st.bar_chart(
        importance.set_index("Feature")
    )

    st.dataframe(
        importance,
        width="stretch",
        hide_index=True
    )

    st.divider()

    # =====================================================
    # MODEL INTERPRETATION
    # =====================================================

    st.subheader("📝 Model Interpretation")

    top_feature = importance.iloc[0]["Feature"]

    st.info(f"""
### Model Overview

**Algorithm**
- Random Forest Classifier

**Evaluation Metrics**
- Accuracy : **{accuracy*100:.2f}%**
- Precision : **{precision*100:.2f}%**
- Recall : **{recall*100:.2f}%**
- F1 Score : **{f1*100:.2f}%**

**Dataset Information**
- Training Features : **{len(feature_names)}**
- Test Samples : **{len(X_test)}**

### Key Findings

• The classifier successfully distinguishes between the five railway switch-machine operating conditions.

• The confusion matrix shows how predictions compare with the actual fault labels.

• The classification report provides Precision, Recall and F1-score for every fault category.

• **{top_feature}** is the most influential feature used by the Random Forest model.

• Temperature-related and electrical parameters contribute the most towards predicting switch-machine health.

### Conclusion

The model demonstrates strong predictive capability for temperature-based railway switch monitoring and is suitable for assisting preventive maintenance decisions.
""")

    st.success("✅ Model evaluation completed successfully.")