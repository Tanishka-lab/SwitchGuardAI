"""
=========================================================
SwitchGuardAI Dashboard
=========================================================

Main dashboard entry point.

Responsible only for:
1. Page configuration
2. Sidebar navigation
3. Loading different dashboard pages

Author : Tanishka
Project : SwitchGuardAI
"""

import streamlit as st
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from components.home import show_home
from components.prediction import show_prediction
from components.performance import show_performance
from components.explainability import show_explainability
from components.report import show_report
from components.about import show_about


# -------------------------------------------------
# Streamlit configuration
# -------------------------------------------------

st.set_page_config(
    page_title="SwitchGuardAI",
    page_icon="🚆",
    layout="wide"
)

# -------------------------------------------------
# Sidebar
# -------------------------------------------------

st.sidebar.title("🚆 SwitchGuardAI")

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Home",
        "🎯 Prediction",
        "📊 Model Performance",
        "🔍 Explainability",
        "📄 Report Generation",
        "ℹ About"
    ]
)

# -------------------------------------------------
# Route pages
# -------------------------------------------------

if page == "🏠 Home":
    show_home()

elif page == "🎯 Prediction":
    show_prediction()

elif page == "📊 Model Performance":
    show_performance()

elif page == "🔍 Explainability":
    show_explainability()

elif page == "📄 Report Generation":
    show_report()

elif page == "ℹ About":
    show_about()