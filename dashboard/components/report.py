"""
=========================================================
SwitchGuardAI - Dashboard Report Generation Component
=========================================================

Reads the LAST prediction made on the Prediction page
(saved into st.session_state), lets the user assign an
Asset ID, previews the report contents, and generates a
downloadable PDF using reporting/report_generator.py.

This file only handles DISPLAY and USER INPUT - all actual
PDF-building and recommendation logic lives in the shared
reporting/ package, so both this page and prediction.py's
on-screen panel always stay consistent with each other.

Author : Tanishka
Project : SwitchGuardAI
"""

import streamlit as st

from reporting.report_generator import generate_report
from reporting.maintenance_recommendation import get_recommendation

FAULT_COLORS = {
    "Normal": "🟢",
    "Maintenance_Overdue": "🟡",
    "Mechanical_Resistance": "🟠",
    "Overheat": "🟠",
    "Critical": "🔴"
}


def estimate_health_index(prediction, confidence, probabilities=None):
    """
    Estimates a 0-100 health index for display in the report.

    If the full probability breakdown is available (stored as
    st.session_state["probabilities"]), uses P(Normal)*100 as a
    natural health estimate - the more likely the model thinks
    the switch is Normal, the healthier it is judged to be.

    Otherwise falls back to a simple heuristic based on the
    predicted class and its confidence.
    """

    if probabilities and "Normal" in probabilities:
        return probabilities["Normal"] * 100

    if prediction == "Normal":
        return confidence

    return max(0.0, 100 - confidence)


def show_report():
    st.title("📄 Report Generation")

    st.write(
        "Generate a downloadable PDF health report based on the "
        "most recent prediction made on the Prediction page."
    )

    prediction = st.session_state.get("prediction")
    confidence = st.session_state.get("confidence")
    sensor_data = st.session_state.get("sensor_data")
    probabilities = st.session_state.get("probabilities")

    if prediction is None:
        st.warning(
            "⚠️ No prediction found yet. Please go to the "
            "**Prediction** page first, run a prediction "
            "(Quick Demo, Manual Entry, or CSV), then come back here."
        )
        return

    health_index = estimate_health_index(prediction, confidence, probabilities)

    st.divider()
    st.subheader("Asset Details")

    switch_id = st.text_input(
        "Asset ID",
        value="SW-MANUAL-01",
        help="Enter the switch machine ID this report is for."
    )

    st.divider()
    st.subheader("Report Preview")

    icon = FAULT_COLORS.get(prediction, "⚪")
    recommendation = get_recommendation(prediction)

    col1, col2, col3 = st.columns(3)
    col1.metric("Fault Status", f"{icon} {prediction}")
    col2.metric("Estimated Health Index", f"{health_index:.1f} / 100")
    col3.metric("Risk Level", recommendation["risk_level"])

    st.write(f"**Next Maintenance Date:** {recommendation['next_maintenance_date']}")
    st.write(f"**Summary:** {recommendation['summary']}")

    if sensor_data:
        st.write("**Sensor Readings included in report:**")
        st.json(sensor_data)

    st.divider()

    if st.button("🧾 Generate PDF Report", type="primary"):

        readable_sensor_data = None
        if sensor_data:
            readable_sensor_data = {k: v for k, v in sensor_data.items()}

        output_path = generate_report(
            switch_id=switch_id,
            fault_label=prediction,
            health_index=health_index,
            sensor_readings=readable_sensor_data
        )

        with open(output_path, "rb") as f:
            pdf_bytes = f.read()

        st.success(f"Report generated for {switch_id}.")

        st.download_button(
            label="📥 Download PDF Report",
            data=pdf_bytes,
            file_name=f"{switch_id}_report.pdf",
            mime="application/pdf"
        )