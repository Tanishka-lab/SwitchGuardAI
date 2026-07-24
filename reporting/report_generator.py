"""
=========================================================
SwitchGuardAI - PDF Report Generator
=========================================================

Generates a PDF health report for a single switch, containing:
- Asset ID
- Health Index
- Fault Label
- Risk Level
- Recommendation
- Next Maintenance Date

Uses the SHARED recommendation logic from
reporting/maintenance_recommendation.py, so this PDF and the
dashboard's on-screen "Recommended Action" panel always stay
consistent with each other.

Author : Tanishka
Project : SwitchGuardAI
"""

import os
from datetime import datetime

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

from reporting.maintenance_recommendation import get_recommendation

REPORTS_DIR = "reports"

RISK_COLORS = {
    "Low": colors.HexColor("#2e7d32"),
    "Moderate": colors.HexColor("#f9a825"),
    "Moderate-High": colors.HexColor("#ef6c00"),
    "High": colors.HexColor("#d84315"),
    "Severe": colors.HexColor("#c62828"),
}


def generate_report(
    switch_id,
    fault_label,
    health_index,
    sensor_readings=None,
    output_path=None
):
    """
    Builds a PDF health report for one switch.

    Parameters
    ----------
    switch_id : str
    fault_label : str
    health_index : float
    sensor_readings : dict, optional
    output_path : str, optional
        Defaults to reports/<switch_id>_report.pdf

    Returns
    -------
    output_path : str
    """

    recommendation = get_recommendation(fault_label)

    if output_path is None:
        os.makedirs(REPORTS_DIR, exist_ok=True)
        output_path = os.path.join(REPORTS_DIR, f"{switch_id}_report.pdf")

    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        topMargin=20 * mm,
        bottomMargin=20 * mm
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "TitleStyle",
        parent=styles["Title"],
        textColor=colors.HexColor("#1a237e")
    )

    risk_color = RISK_COLORS.get(recommendation["risk_level"], colors.black)

    status_style = ParagraphStyle(
        "StatusStyle",
        parent=styles["Heading1"],
        textColor=risk_color
    )

    story = []

    story.append(Paragraph("SwitchGuardAI - Health Report", title_style))
    story.append(Paragraph(
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        styles["Normal"]
    ))
    story.append(Spacer(1, 10 * mm))

    summary_data = [
        ["Asset ID", switch_id],
        ["Fault Status", fault_label],
        ["Health Index", f"{health_index:.1f} / 100"],
        ["Risk Level", recommendation["risk_level"]],
        ["Next Maintenance Date", recommendation["next_maintenance_date"]],
    ]

    summary_table = Table(summary_data, colWidths=[60 * mm, 90 * mm])
    summary_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#e8eaf6")),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("TEXTCOLOR", (1, 1), (1, 1), risk_color),
        ("FONTNAME", (1, 1), (1, 1), "Helvetica-Bold"),
    ]))

    story.append(summary_table)
    story.append(Spacer(1, 8 * mm))

    story.append(Paragraph(fault_label.upper().replace("_", " "), status_style))
    story.append(Paragraph(recommendation["summary"], styles["Normal"]))
    story.append(Spacer(1, 6 * mm))

    if sensor_readings:
        story.append(Paragraph("Current Sensor Readings", styles["Heading2"]))

        reading_rows = [[k, str(v)] for k, v in sensor_readings.items()]
        reading_table = Table(reading_rows, colWidths=[70 * mm, 80 * mm])
        reading_table.setStyle(TableStyle([
            ("GRID", (0, 0), (-1, -1), 0.4, colors.lightgrey),
            ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]))
        story.append(reading_table)
        story.append(Spacer(1, 8 * mm))

    story.append(Paragraph("Maintenance Recommendation", styles["Heading2"]))

    for action in recommendation["actions"]:
        story.append(Paragraph(f"• {action}", styles["Normal"]))

    story.append(Spacer(1, 10 * mm))
    story.append(Paragraph(
        "This is a system-generated report from the SwitchGuardAI "
        "predictive maintenance prototype. Recommendations should be "
        "verified by a qualified maintenance engineer before action.",
        styles["Italic"]
    ))

    doc.build(story)

    return output_path


if __name__ == "__main__":

    path = generate_report(
        switch_id="SW001",
        fault_label="Critical",
        health_index=18.5,
        sensor_readings={
            "Ambient Temperature": "30.0 C",
            "Motor Temperature": "120.0 C",
            "Motor Current": "26.00 A",
            "Operation Duration": "8.00 s",
            "Days Since Maintenance": 90,
        }
    )

    print(f"Report generated at: {path}")