"""
=========================================================
SwitchGuardAI - Maintenance Recommendation Logic
=========================================================

Single, shared source of truth for maintenance recommendations,
risk levels, and next-maintenance-date calculations.

Used by BOTH:
- dashboard/components/prediction.py (for the on-screen
  "Recommended Action" panel)
- reporting/report_generator.py (for the PDF report)

Keeping this logic in ONE file (instead of duplicating similar
text in both places) means an update here automatically applies
everywhere - no risk of the dashboard and PDF drifting out of
sync over time.

Author : Tanishka
Project : SwitchGuardAI
"""

from datetime import datetime, timedelta


RECOMMENDATIONS = {

    "Normal": {
        "risk_level": "Low",
        "summary": "The switch machine is operating within expected limits.",
        "actions": [
            "Continue normal operation.",
            "No abnormal thermal behaviour detected.",
            "Continue routine temperature monitoring.",
            "Perform scheduled preventive maintenance only."
        ],
        "next_maintenance_days": 90
    },

    "Maintenance_Overdue": {
        "risk_level": "Moderate",
        "summary": "The machine is functioning, but scheduled maintenance is overdue.",
        "actions": [
            "Schedule preventive maintenance immediately.",
            "Inspect lubrication of moving components.",
            "Verify gearbox wear.",
            "Reset maintenance schedule after servicing.",
            "Continue monitoring temperatures until maintenance is completed."
        ],
        "next_maintenance_days": 30
    },

    "Mechanical_Resistance": {
        "risk_level": "Moderate-High",
        "summary": "Increased mechanical resistance detected during switch operation.",
        "actions": [
            "Inspect switch mechanism for obstruction.",
            "Check gearbox alignment.",
            "Inspect moving rails and actuator.",
            "Verify excessive motor load.",
            "Re-test switch operation after inspection."
        ],
        "next_maintenance_days": 14
    },

    "Overheat": {
        "risk_level": "High",
        "summary": "Abnormal temperature behaviour detected.",
        "actions": [
            "Inspect motor cooling.",
            "Check electrical loading.",
            "Verify ventilation around the switch machine.",
            "Monitor motor temperature before returning to service.",
            "Inspect temperature sensors for abnormal readings."
        ],
        "next_maintenance_days": 7
    },

    "Critical": {
        "risk_level": "Severe",
        "summary": "Critical operating condition detected.",
        "actions": [
            "Stop further operation if possible.",
            "Dispatch maintenance personnel immediately.",
            "Inspect motor, gearbox and electrical system.",
            "Check for thermal damage.",
            "Perform complete inspection before returning the switch to service."
        ],
        "next_maintenance_days": 1
    }
}


def get_recommendation(fault_label):
    """
    Returns the full recommendation package for a given fault label.

    Returns
    -------
    dict with keys: risk_level, summary, actions (list), next_maintenance_date (str)
    """

    info = RECOMMENDATIONS.get(fault_label, RECOMMENDATIONS["Normal"])

    next_date = datetime.now() + timedelta(days=info["next_maintenance_days"])

    return {
        "risk_level": info["risk_level"],
        "summary": info["summary"],
        "actions": info["actions"],
        "next_maintenance_date": next_date.strftime("%Y-%m-%d")
    }


if __name__ == "__main__":

    for label in RECOMMENDATIONS.keys():
        result = get_recommendation(label)
        print(f"--- {label} ---")
        print("Risk Level:", result["risk_level"])
        print("Summary:", result["summary"])
        print("Next Maintenance Date:", result["next_maintenance_date"])
        print("Actions:")
        for a in result["actions"]:
            print("  -", a)
        print()