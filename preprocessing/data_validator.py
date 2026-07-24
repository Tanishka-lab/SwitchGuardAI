"""
=========================================================
SwitchGuardAI Data Validator
=========================================================

Validates the raw dataset before preprocessing.

Checks:
1. Missing values
2. Duplicate rows
3. Timestamp ordering
4. Numeric range validation
5. Invalid fault labels

Author : Tanishka
Project : SwitchGuardAI
=========================================================
"""

import pandas as pd

from preprocessing.data_loader import DataLoader

from configs.sensor_specifications import (
    MAX_MOTOR_TEMP,
    MAX_GEARBOX_TEMP,
    MAX_LOCK_TEMP,
    MAX_CABINET_TEMP,
    MAX_CURRENT,
    MAX_OPERATION_DURATION
)


VALID_FAULTS = [
    "Normal",
    "Maintenance_Overdue",
    "Mechanical_Resistance",
    "Overheat",
    "Critical"
]


class DataValidator:

    def __init__(self, dataframe):
        self.df = dataframe

    # --------------------------------------------------
    # Missing Values
    # --------------------------------------------------

    def check_missing_values(self):

        missing = self.df.isnull().sum()

        print("\n" + "=" * 60)
        print("MISSING VALUES")
        print("=" * 60)
        print(missing)

    # --------------------------------------------------
    # Duplicate Rows
    # --------------------------------------------------

    def check_duplicates(self):

        duplicates = self.df.duplicated().sum()

        print("\n" + "=" * 60)
        print("DUPLICATE ROWS")
        print("=" * 60)
        print(duplicates)

    # --------------------------------------------------
    # Timestamp Order
    # --------------------------------------------------

    def check_timestamp_order(self):

        ordered = self.df["Timestamp"].is_monotonic_increasing

        print("\n" + "=" * 60)
        print("TIMESTAMP ORDER")
        print("=" * 60)

        if ordered:
            print("PASS : Dataset is time ordered.")
        else:
            print("WARNING : Dataset is NOT time ordered.")

    # --------------------------------------------------
    # Numeric Ranges
    # --------------------------------------------------

    def check_ranges(self):

        print("\n" + "=" * 60)
        print("NUMERIC RANGE CHECK")
        print("=" * 60)

        checks = {

            "Motor Temperature":
            (self.df["Motor_Temperature"] <= MAX_MOTOR_TEMP).all(),

            "Gearbox Temperature":
            (self.df["Gearbox_Temperature"] <= MAX_GEARBOX_TEMP).all(),

            "Lock Temperature":
            (self.df["Lock_Temperature"] <= MAX_LOCK_TEMP).all(),

            "Cabinet Temperature":
            (self.df["Control_Cabinet_Temperature"] <= MAX_CABINET_TEMP).all(),

            "Motor Current":
            (self.df["Motor_Current"] <= MAX_CURRENT).all(),

            "Operation Duration":
            (self.df["Operation_Duration"] <= MAX_OPERATION_DURATION).all(),

            "Health Index":
            (
                (self.df["Health_Index"] >= 0) &
                (self.df["Health_Index"] <= 100)
            ).all()

        }

        for item, result in checks.items():

            status = "PASS" if result else "FAIL"

            print(f"{item:<30} {status}")

    # --------------------------------------------------
    # Fault Labels
    # --------------------------------------------------

    def check_fault_labels(self):

        invalid = self.df[
            ~self.df["Fault_Label"].isin(VALID_FAULTS)
        ]

        print("\n" + "=" * 60)
        print("FAULT LABEL VALIDATION")
        print("=" * 60)

        if len(invalid) == 0:
            print("PASS : All fault labels are valid.")
        else:
            print(f"FAIL : {len(invalid)} invalid rows found.")

    # --------------------------------------------------
    # Full Validation
    # --------------------------------------------------

    def validate(self):

        self.check_missing_values()

        self.check_duplicates()

        self.check_timestamp_order()

        self.check_ranges()

        self.check_fault_labels()

        print("\n" + "=" * 60)
        print("DATA VALIDATION COMPLETE")
        print("=" * 60)


# =====================================================
# MAIN
# =====================================================

if __name__ == "__main__":

    loader = DataLoader()

    dataset = loader.load_data()

    validator = DataValidator(dataset)

    validator.validate()