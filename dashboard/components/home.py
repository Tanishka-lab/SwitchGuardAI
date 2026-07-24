import streamlit as st
import joblib
import pandas as pd

from sklearn.metrics import accuracy_score

MODEL_PATH = "saved_models/random_forest.pkl"
X_TEST_PATH = "data/processed/X_test.csv"
Y_TEST_PATH = "data/processed/y_test.csv"


def get_model_accuracy():

    model = joblib.load(MODEL_PATH)

    X_test = pd.read_csv(X_TEST_PATH)

    y_test = pd.read_csv(Y_TEST_PATH).squeeze()

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    return f"{accuracy*100:.2f}%"

def show_home():

    st.title("🚦 SwitchGuardAI")

    st.subheader(
        "Temperature-Based Predictive Maintenance for Railway Switch Machines"
    )

    st.divider()

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Model",
            "Random Forest"
        )

    with col2:
        st.metric(
            "Accuracy",
             get_model_accuracy()
        )

    with col3:
        st.metric(
            "Features",
            "38"
        )

    with col4:
        st.metric(
            "Fault Classes",
            "5"
        )

    st.divider()

    st.header("Project Overview")

    st.write(
        """
SwitchGuardAI predicts faults in railway switch machines using
temperature, electrical and engineered maintenance features.

The goal is to identify failures before they occur so that
maintenance teams can take preventive action instead of reacting
after equipment breakdown.
"""
    )

    st.divider()

    st.header("Dashboard Modules")

    c1, c2 = st.columns(2)

    with c1:

        st.success("🎯 Prediction")

        st.success("📊 Model Performance")

        st.success("🔍 Explainability")

    with c2:

        st.success("📄 Report Generation")

        st.success("ℹ About")

    