"""
=========================================================
SwitchGuardAI Data Cleaner
=========================================================

Cleans the raw dataset before feature engineering.

Responsibilities
----------------
1. Remove duplicates
2. Sort by timestamp
3. Reset index
4. Standardize numeric precision
5. Save cleaned dataset

Author : Tanishka
Project : SwitchGuardAI
=========================================================
"""

import os

from preprocessing.data_loader import DataLoader


OUTPUT_PATH = "data/processed/switch_machine_dataset_clean.csv"


class DataCleaner:

    def __init__(self, dataframe):
        self.df = dataframe.copy()

    # --------------------------------------------------
    # Remove Duplicate Rows
    # --------------------------------------------------

    def remove_duplicates(self):

        before = len(self.df)

        self.df = self.df.drop_duplicates()

        removed = before - len(self.df)

        print(f"Duplicates Removed : {removed}")

    # --------------------------------------------------
    # Sort Dataset
    # --------------------------------------------------

    def sort_dataset(self):

        self.df = self.df.sort_values(
            by="Timestamp"
        ).reset_index(drop=True)

        print("Dataset sorted by Timestamp.")

    # --------------------------------------------------
    # Standardize Numeric Precision
    # --------------------------------------------------

    def round_numeric_columns(self):

        numeric_columns = [

            "Ambient_Temperature",

            "Motor_Temperature",

            "Gearbox_Temperature",

            "Lock_Temperature",

            "Control_Cabinet_Temperature",

            "Motor_Current",

            "Operation_Duration",

            "Days_Since_Last_Maintenance",

            "Health_Index"

        ]

        self.df[numeric_columns] = (
            self.df[numeric_columns]
            .round(2)
        )

        print("Numeric precision standardized.")

    # --------------------------------------------------
    # Save Clean Dataset
    # --------------------------------------------------

    def save_dataset(self):

        os.makedirs("data/processed", exist_ok=True)

        self.df.to_csv(
            OUTPUT_PATH,
            index=False
        )

        print(f"\nSaved cleaned dataset to:\n{OUTPUT_PATH}")

    # --------------------------------------------------
    # Complete Cleaning Pipeline
    # --------------------------------------------------

    def clean(self):

        print("=" * 60)
        print("DATA CLEANING")
        print("=" * 60)

        self.remove_duplicates()

        self.sort_dataset()

        self.round_numeric_columns()

        self.save_dataset()

        print("\nCleaning Complete.")

        return self.df


# =====================================================
# MAIN
# =====================================================

if __name__ == "__main__":

    loader = DataLoader()

    dataset = loader.load_data()

    cleaner = DataCleaner(dataset)

    cleaned_dataset = cleaner.clean()

    print("\nShape :", cleaned_dataset.shape)

    print("\nFirst Five Rows")

    print(cleaned_dataset.head())