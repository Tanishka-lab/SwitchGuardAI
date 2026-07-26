import streamlit as st


def show_about():

    st.title("🚆 About SwitchGuardAI")
    st.markdown("---")

    st.info("""
**SwitchGuardAI** is an AI-powered predictive maintenance system designed to
monitor the health of railway switch machines.

The system analyzes operational data using Machine Learning to predict failures
before they occur, helping reduce downtime, improve safety, and optimize
maintenance schedules.
""")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Health Score", "0 - 100")

    with col2:
        st.metric("Fault Types", "5")

    with col3:
        st.metric("Reports", "Automatic")

    st.markdown("---")

    st.subheader("🎯 Project Objectives")

    st.markdown("""
- Predict switch machine failures
- Estimate equipment health
- Detect fault categories
- Recommend maintenance actions
- Generate health reports
- Monitor railway assets through an interactive dashboard
""")

    st.subheader("🤖 Machine Learning Features")

    st.markdown("""
The model uses the following inputs:

- Ambient Temperature
- Motor Temperature
- Gearbox Temperature
- Lock Temperature
- Control Cabinet Temperature
- Motor Current
- Operation Duration
- Days Since Last Maintenance
- Previous Health Index
- Season
- Location Type

The model predicts:

- Health Index
- Fault Category
- Risk Level
""")

    st.subheader("⚙️ Technology Stack")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
**Backend**
- Python
- Scikit-learn
- Pandas
- NumPy
- Joblib
""")

    with col2:
        st.markdown("""
**Frontend**
- Streamlit
- Matplotlib
- ReportLab
""")

    st.subheader("📁 Project Modules")

    st.markdown("""
- Data Generation
- Feature Engineering
- Model Training
- Prediction Engine
- Maintenance Recommendation
- Report Generator
- Dashboard
""")

    st.subheader("🚄 Benefits")

    st.markdown("""
- Early fault detection
- Reduced maintenance cost
- Improved railway safety
- Increased asset reliability
- Data-driven maintenance planning
""")

    st.subheader("👩‍💻 Developer")

    st.success("""
**Developer:** Tanishka Trehan

**Project:** SwitchGuardAI

**Technology:** Python • Machine Learning • Streamlit
""")

    st.markdown("---")
    st.caption("© 2026 SwitchGuardAI | AI-Based Railway Predictive Maintenance System")
