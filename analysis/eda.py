"""
=========================================================
SwitchGuardAI - Exploratory Data Analysis (EDA)
=========================================================

This script performs an initial analysis of the generated
synthetic railway switch dataset.

Tasks:
1. Dataset Overview
2. Missing Values
3. Duplicate Check
4. Statistical Summary
5. Fault Distribution
6. Health Index Distribution
7. Temperature Distributions
8. Correlation Matrix
9. Scatter Plots
10. Save Correlation Matrix

Author : Tanishka
Project : SwitchGuardAI
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# =====================================================
# LOAD DATASET
# =====================================================

DATA_PATH = "data/raw/switch_machine_dataset.csv"

df = pd.read_csv(DATA_PATH)

os.makedirs("analysis/plots", exist_ok=True)

# =====================================================
# BASIC INFORMATION
# =====================================================

print("=" * 60)
print("DATASET SHAPE")
print("=" * 60)

print(df.shape)

print("\n")

print("=" * 60)
print("COLUMNS")
print("=" * 60)

print(df.columns.tolist())

print("\n")

print("=" * 60)
print("DATA TYPES")
print("=" * 60)

print(df.dtypes)

print("\n")

print("=" * 60)
print("MISSING VALUES")
print("=" * 60)

print(df.isnull().sum())

print("\n")

print("=" * 60)
print("DUPLICATE ROWS")
print("=" * 60)

print(df.duplicated().sum())

print("\n")

print("=" * 60)
print("STATISTICAL SUMMARY")
print("=" * 60)

print(df.describe())

# =====================================================
# FAULT DISTRIBUTION
# =====================================================

print("\n")
print("=" * 60)
print("FAULT DISTRIBUTION")
print("=" * 60)

print(df["Fault_Label"].value_counts())

plt.figure(figsize=(8,5))

df["Fault_Label"].value_counts().plot(kind="bar")

plt.title("Fault Distribution")

plt.xlabel("Fault")

plt.ylabel("Count")

plt.tight_layout()

plt.savefig("analysis/plots/fault_distribution.png")

plt.show()

# =====================================================
# HEALTH INDEX
# =====================================================

plt.figure(figsize=(8,5))

plt.hist(df["Health_Index"], bins=40)

plt.title("Health Index Distribution")

plt.xlabel("Health Index")

plt.ylabel("Frequency")

plt.tight_layout()

plt.savefig("analysis/plots/health_distribution.png")

plt.show()

# =====================================================
# TEMPERATURE DISTRIBUTIONS
# =====================================================

temperature_columns = [

    "Ambient_Temperature",

    "Motor_Temperature",

    "Gearbox_Temperature",

    "Lock_Temperature",

    "Control_Cabinet_Temperature"

]

for column in temperature_columns:

    plt.figure(figsize=(8,4))

    plt.hist(df[column], bins=40)

    plt.title(column)

    plt.xlabel("Temperature (°C)")

    plt.ylabel("Frequency")

    plt.tight_layout()

    filename = column.lower().replace(" ", "_")

    plt.savefig(f"analysis/plots/{filename}.png")

    plt.show()

# =====================================================
# LOCATION DISTRIBUTION
# =====================================================

plt.figure(figsize=(6,4))

df["Location_Type"].value_counts().plot(kind="bar")

plt.title("Location Type Distribution")

plt.xlabel("Location")

plt.ylabel("Count")

plt.tight_layout()

plt.savefig("analysis/plots/location_distribution.png")

plt.show()

# =====================================================
# SEASON DISTRIBUTION
# =====================================================

plt.figure(figsize=(6,4))

df["Season"].value_counts().plot(kind="bar")

plt.title("Season Distribution")

plt.xlabel("Season")

plt.ylabel("Count")

plt.tight_layout()

plt.savefig("analysis/plots/season_distribution.png")

plt.show()

# =====================================================
# CORRELATION MATRIX
# =====================================================

numeric_df = df.select_dtypes(include=np.number)

corr = numeric_df.corr()

print("\n")

print("=" * 60)
print("CORRELATION MATRIX")
print("=" * 60)

print(corr)

corr.to_csv("analysis/correlation_matrix.csv")

# =====================================================
# SCATTER PLOTS
# =====================================================

plt.figure(figsize=(7,5))

plt.scatter(

    df["Motor_Temperature"],

    df["Health_Index"],

    s=2

)

plt.xlabel("Motor Temperature")

plt.ylabel("Health Index")

plt.title("Motor Temperature vs Health Index")

plt.tight_layout()

plt.savefig("analysis/plots/motor_temp_vs_health.png")

plt.show()

# -----------------------------------------------------

plt.figure(figsize=(7,5))

plt.scatter(

    df["Motor_Current"],

    df["Operation_Duration"],

    s=2

)

plt.xlabel("Motor Current (A)")

plt.ylabel("Operation Duration (s)")

plt.title("Motor Current vs Operation Duration")

plt.tight_layout()

plt.savefig("analysis/plots/current_vs_duration.png")

plt.show()

# =====================================================
# HEALTH CLASS SUMMARY
# =====================================================

print("\n")

print("=" * 60)
print("HEALTH INDEX")
print("=" * 60)

print(df["Health_Index"].describe())

print("\nEDA COMPLETED SUCCESSFULLY!")

print("\nPlots saved to:")

print("analysis/plots/")