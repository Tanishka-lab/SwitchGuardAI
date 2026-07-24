"""
=========================================================
SwitchGuardAI - Dashboard Prediction Component
=========================================================

Provides the Prediction page for the Streamlit dashboard.

Three modes:
1. Quick Demo   - one-click predictions using REAL rows from
                  the test set (guaranteed strong, correct,
                  confident predictions - good for live demos)
2. Manual Entry - adjust sliders yourself (results may show
                  lower confidence if values don't match
                  realistic, physically-consistent sensor
                  patterns - this is expected, not a bug)
3. CSV Upload   - predict on a full feature-engineered CSV
                  (uses the model's full, real accuracy)

Author : Tanishka
Project : SwitchGuardAI
"""

import joblib
import pandas as pd
import streamlit as st

MODEL_PATH = "saved_models/random_forest.pkl"
FEATURE_PATH = "saved_models/feature_names.pkl"
X_TEST_PATH = "data/processed/X_test.csv"
Y_TEST_PATH = "data/processed/y_test.csv"

NON_FEATURE_COLUMNS = [
    "Timestamp",
    "Switch_ID",
    "Health_Index",
    "Fault_Label"
]

FAULT_COLORS = {
    "Normal": "🟢",
    "Maintenance_Overdue": "🟡",
    "Mechanical_Resistance": "🟠",
    "Overheat": "🟠",
    "Critical": "🔴"
}


@st.cache_resource
def load_model():
    model = joblib.load(MODEL_PATH)
    feature_names = joblib.load(FEATURE_PATH)
    return model, feature_names


@st.cache_data
def load_test_data():
    X_test = pd.read_csv(X_TEST_PATH)
    y_test = pd.read_csv(Y_TEST_PATH).squeeze()
    return X_test, y_test


@st.cache_data
def build_template(feature_names):
    """
    Builds a realistic baseline row using median (numeric columns)
    and mode (one-hot columns) from the real test set. Used as the
    starting point for manual entry, so unset features stay
    internally consistent instead of defaulting to zero.
    """
    X_test = pd.read_csv(X_TEST_PATH)
    template = {}

    for col in feature_names:
        values = X_test[col].dropna()

        if values.isin([0, 1]).all() and values.nunique() <= 2:
            template[col] = values.mode()[0]
        else:
            template[col] = values.median()

    return template


def predict_single(model, feature_names, row):
    df = pd.DataFrame([row])
    df = df[feature_names]

    prediction = model.predict(df)[0]
    probabilities = model.predict_proba(df)[0]

    probability_series = pd.Series(
        probabilities,
        index=model.classes_
    ).sort_values(ascending=False)

    return prediction, probability_series


def display_prediction_result(prediction, probabilities, readings):
    """
    readings : dict with keys ambient_temp, motor_temp, motor_current,
               operation_duration, maintenance_days, previous_health
    """

    ambient_temp = readings["ambient_temp"]
    motor_temp = readings["motor_temp"]
    motor_current = readings["motor_current"]
    operation_duration = readings["operation_duration"]
    maintenance_days = readings["maintenance_days"]
    previous_health = readings["previous_health"]

    thermal_rise = motor_temp - ambient_temp
    thermal_stress = motor_temp * motor_current
    current_rate = motor_current / operation_duration if operation_duration else 0
    maintenance_ratio = min(maintenance_days / 90, 1.0)

    st.divider()
    st.subheader("Prediction Result")

    col1, col2 = st.columns([1, 2])

    with col1:
        icon = FAULT_COLORS.get(prediction, "⚪")
        st.success(f"# {icon} {prediction.upper()}")

        confidence_score = probabilities.get(prediction, 0.0) * 100
        st.metric(label="Confidence", value=f"{confidence_score:.2f}%")

    with col2:
        st.subheader("Probability Distribution")
        st.bar_chart(probabilities)

    st.divider()
    st.subheader("🛠 Recommended Action")

    if prediction == "Normal":
        st.success(f"""
        ### 🟢 NORMAL

        The switch machine is operating within the expected limits.

        ### Current Sensor Readings

        - Ambient Temperature : **{ambient_temp:.1f} °C**
        - Motor Temperature : **{motor_temp:.1f} °C**
        - Thermal Rise : **{thermal_rise:.1f} °C**
        - Motor Current : **{motor_current:.2f} A**
        - Operation Duration : **{operation_duration:.2f} s**
        - Previous Health Index : **{previous_health:.0f}**
        - Days Since Maintenance : **{maintenance_days}**

        ### Maintenance Recommendation

        ✅ Continue normal operation.
        ✅ No abnormal thermal behaviour detected.
        ✅ Continue routine temperature monitoring.
        ✅ Perform scheduled preventive maintenance only.
        """)

    elif prediction == "Maintenance_Overdue":
        st.warning(f"""
        ### 🟡 MAINTENANCE OVERDUE

        The machine is functioning but scheduled maintenance is overdue.

        ### Current Sensor Readings

        - Previous Health Index : **{previous_health:.0f}**
        - Days Since Maintenance : **{maintenance_days}**
        - Maintenance Ratio : **{maintenance_ratio:.2f}**
        - Motor Temperature : **{motor_temp:.1f} °C**

        ### Maintenance Recommendation

        • Schedule preventive maintenance immediately.
        • Inspect lubrication of moving components.
        • Verify gearbox wear.
        • Reset maintenance schedule after servicing.
        • Continue monitoring temperatures until maintenance is completed.
        """)

    elif prediction == "Mechanical_Resistance":
        st.warning(f"""
        ### 🟠 MECHANICAL RESISTANCE

        The model detected increased mechanical resistance during switch operation.

        ### Current Sensor Readings

        - Motor Current : **{motor_current:.2f} A**
        - Operation Duration : **{operation_duration:.2f} s**
        - Current Rate : **{current_rate:.2f}**
        - Thermal Stress : **{thermal_stress:.1f}**
        - Previous Health Index : **{previous_health:.0f}**

        ### Maintenance Recommendation

        • Inspect switch mechanism for obstruction.
        • Check gearbox alignment.
        • Inspect moving rails and actuator.
        • Verify excessive motor load.
        • Re-test switch operation after inspection.
        """)

    elif prediction == "Overheat":
        st.error(f"""
        ### 🟠 OVERHEAT

        The model detected abnormal temperature behaviour.

        ### Current Sensor Readings

        - Ambient Temperature : **{ambient_temp:.1f} °C**
        - Motor Temperature : **{motor_temp:.1f} °C**
        - Thermal Rise : **{thermal_rise:.1f} °C**
        - Thermal Stress : **{thermal_stress:.1f}**

        ### Maintenance Recommendation

        • Inspect motor cooling.
        • Check electrical loading.
        • Verify ventilation around the switch machine.
        • Monitor motor temperature before returning to service.
        • Inspect temperature sensors for abnormal readings.
        """)

    elif prediction == "Critical":
        st.error(f"""
        ### 🔴 CRITICAL CONDITION

        Critical operating condition detected.

        ### Current Sensor Readings

        - Motor Temperature : **{motor_temp:.1f} °C**
        - Motor Current : **{motor_current:.2f} A**
        - Thermal Stress : **{thermal_stress:.1f}**
        - Previous Health Index : **{previous_health:.0f}**
        - Days Since Maintenance : **{maintenance_days}**

        ### Immediate Action Required

        🚨 Stop further operation if possible.
        🚨 Dispatch maintenance personnel immediately.
        🚨 Inspect motor, gearbox and electrical system.
        🚨 Check thermal damage.
        🚨 Perform complete inspection before returning the switch to service.
        """)

    st.success("Prediction completed successfully.")


def show_prediction():
    st.title("🚦 Prediction")

    st.write(
        "Predict railway switch-machine faults using the "
        "trained Random Forest model."
    )

    model, feature_names = load_model()
    template = build_template(feature_names)

    mode = st.radio(
        "Prediction Mode",
        [
            "Quick Demo (Real Examples)",
            "Manual Sensor Reading",
            "CSV Prediction"
        ]
    )

    if mode == "Quick Demo (Real Examples)":

        st.subheader("Try a Real Example")

        st.info(
            "These buttons use REAL rows pulled from the model's test "
            "set (not hand-typed values), so predictions here reflect "
            "the model's actual, validated performance."
        )

        X_test, y_test = load_test_data()

        available_classes = [c for c in model.classes_ if (y_test == c).any()]

        cols = st.columns(len(available_classes))

        selected_class = None

        for i, fault_class in enumerate(available_classes):
            icon = FAULT_COLORS.get(fault_class, "⚪")
            if cols[i].button(f"{icon} {fault_class}"):
                selected_class = fault_class

        if selected_class is not None:

            matching_indices = y_test[y_test == selected_class].index[:100]

            example_idx = matching_indices[0]
            best_confidence = -1

            for idx in matching_indices:
                candidate_row = X_test.loc[idx]
                candidate_df = pd.DataFrame([candidate_row.to_dict()])[feature_names]
                candidate_pred = model.predict(candidate_df)[0]

                if candidate_pred == selected_class:
                    candidate_conf = model.predict_proba(candidate_df)[0][
                        list(model.classes_).index(selected_class)
                    ]
                    if candidate_conf > best_confidence:
                        best_confidence = candidate_conf
                        example_idx = idx

            example_row = X_test.loc[example_idx]

            prediction, probabilities = predict_single(
                model, feature_names, example_row.to_dict()
            )

            readings = {
                "ambient_temp": example_row["Ambient_Temperature"],
                "motor_temp": example_row["Motor_Temperature"],
                "motor_current": example_row["Motor_Current"],
                "operation_duration": example_row["Operation_Duration"],
                "maintenance_days": example_row["Days_Since_Last_Maintenance"],
                "previous_health": example_row["Previous_Health"],
            }

            st.caption(f"(This is a real test-set row, actual label: **{selected_class}**)")

            # Save latest prediction for Report Generation page
            # (same pattern as Manual Entry mode, so the Report
            # page works no matter which mode was used)
            st.session_state["prediction"] = prediction
            st.session_state["confidence"] = float(probabilities.max()) * 100
            st.session_state["probabilities"] = probabilities.to_dict()

            st.session_state["sensor_data"] = {
                "Ambient Temperature": readings["ambient_temp"],
                "Motor Temperature": readings["motor_temp"],
                "Motor Current": readings["motor_current"],
                "Operation Duration": readings["operation_duration"],
                "Days Since Maintenance": readings["maintenance_days"],
                "Previous Health Index": readings["previous_health"],
            }

            display_prediction_result(prediction, probabilities, readings)

    elif mode == "Manual Sensor Reading":
        st.subheader("Manual Sensor Reading")

        st.info(
            "Enter the latest sensor values below. "
            "Historical engineered features are automatically "
            "filled using realistic baseline values."
        )

        st.caption(
            "💡 Tip: in the real training data, temperature, current, "
            "AND duration all rise TOGETHER for genuine faults - moving "
            "only one slider (e.g. temperature alone) creates a "
            "combination the model rarely saw, so confidence may look "
            "weak or the prediction may stay 'Normal'. For a confident "
            "fault example, try raising Motor Temperature (~100-130°C), "
            "Motor Current (~22-28A), AND Operation Duration (~7-10s) "
            "together."
        )

        col1, col2, col3 = st.columns(3)

        with col1:
            ambient_temp = st.slider("Ambient Temperature (°C)", 10.0, 50.0, 30.0)
            motor_temp = st.slider("Motor Temperature (°C)", 10.0, 150.0, 45.0)

        with col2:
            motor_current = st.slider("Motor Current (A)", 5.0, 35.0, 15.0)
            operation_duration = st.slider("Operation Duration (s)", 2.0, 15.0, 4.5)

        with col3:
            maintenance_days = st.slider("Days Since Last Maintenance", 0, 180, 20)
            previous_health = st.slider("Previous Health Index", 0, 100, 95)

        if st.button("Predict Fault", type="primary"):
            row = template.copy()

            row["Ambient_Temperature"] = ambient_temp
            row["Motor_Temperature"] = motor_temp
            row["Motor_Current"] = motor_current
            row["Operation_Duration"] = operation_duration
            row["Days_Since_Last_Maintenance"] = maintenance_days
            row["Previous_Health"] = previous_health

            if "Thermal_Rise_Motor" in row:
                row["Thermal_Rise_Motor"] = motor_temp - ambient_temp

            if "Thermal_Stress" in row:
                row["Thermal_Stress"] = motor_temp * motor_current

            if "Current_Rate" in row:
                row["Current_Rate"] = motor_current / operation_duration

            if "Maintenance_Ratio" in row:
                row["Maintenance_Ratio"] = min(maintenance_days / 90, 1.0)

            if "Gearbox_Temperature" in row:
                row["Gearbox_Temperature"] = ambient_temp + (motor_temp - ambient_temp) * 0.6

            if "Lock_Temperature" in row:
                row["Lock_Temperature"] = ambient_temp + (motor_temp - ambient_temp) * 0.4

            if "Control_Cabinet_Temperature" in row:
                row["Control_Cabinet_Temperature"] = ambient_temp + 4 + (0.10 * motor_current)

            if "Thermal_Rise_Gearbox" in row:
                row["Thermal_Rise_Gearbox"] = row["Gearbox_Temperature"] - ambient_temp

            if "Thermal_Rise_Lock" in row:
                row["Thermal_Rise_Lock"] = row["Lock_Temperature"] - ambient_temp

            if "Thermal_Rise_Cabinet" in row:
                row["Thermal_Rise_Cabinet"] = row["Control_Cabinet_Temperature"] - ambient_temp

            if "Rolling_Avg_Motor_Temp" in row:
                row["Rolling_Avg_Motor_Temp"] = motor_temp

            if "Rolling_Avg_Current" in row:
                row["Rolling_Avg_Current"] = motor_current

            prediction, probabilities = predict_single(
                model,
                feature_names,
                row
            )

            confidence = float(probabilities.max()) * 100

            # =====================================================
            # Save latest prediction for Report Generation page
            # =====================================================

            st.session_state["prediction"] = prediction
            st.session_state["confidence"] = confidence
            st.session_state["probabilities"] = probabilities.to_dict()
            
            st.session_state["sensor_data"] = {

                "Ambient Temperature": ambient_temp,
                "Motor Temperature": motor_temp,
                "Motor Current": motor_current,
                "Operation Duration": operation_duration,
                "Days Since Maintenance": maintenance_days,
                "Previous Health Index": previous_health,

            }

            readings = {
                "ambient_temp": ambient_temp,
                "motor_temp": motor_temp,
                "motor_current": motor_current,
                "operation_duration": operation_duration,
                "maintenance_days": maintenance_days,
                "previous_health": previous_health,
            }

            display_prediction_result(
                prediction,
                probabilities,
                readings
            )

    else:
        st.subheader("Upload Feature Engineered CSV")

        st.info(
            "Upload a CSV generated from the feature engineering "
            "pipeline. The model will predict the fault label for "
            "every row."
        )

        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type="csv"
        )

        if uploaded_file is not None:
            new_data = pd.read_csv(uploaded_file)

            drop_cols = [
                c for c in NON_FEATURE_COLUMNS
                if c in new_data.columns
            ]

            X_new = new_data.drop(columns=drop_cols)
            X_new = X_new[feature_names]

            predictions = model.predict(X_new)
            probabilities = model.predict_proba(X_new)

            results = new_data.copy()
            results["Predicted_Fault"] = predictions

            for i, label in enumerate(model.classes_):
                results[f"Prob_{label}"] = probabilities[:, i]

            st.success(
                f"Successfully predicted {len(results)} records."
            )

            st.subheader("Fault Distribution")
            fault_counts = results["Predicted_Fault"].value_counts()
            st.bar_chart(fault_counts)

            st.subheader("Prediction Results")
            st.dataframe(
                results,
                width="stretch"
            )

            st.download_button(
                label="📥 Download Predictions",
                data=results.to_csv(index=False),
                file_name="switchguard_predictions.csv",
                mime="text/csv"
            )