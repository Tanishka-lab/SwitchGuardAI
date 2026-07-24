"""
SwitchGuardAI Data Preprocessing Pipeline

Runs:
1. Load Dataset
2. Validate Dataset
3. Clean Dataset
4. Convert Dataset
"""

from preprocessing.data_loader import DataLoader
from preprocessing.data_validator import DataValidator
from preprocessing.data_cleaner import DataCleaner
from preprocessing.dataset_converter import convert_dataset


def main():

    print("=" * 60)
    print("SwitchGuardAI PREPROCESSING PIPELINE")
    print("=" * 60)

    # Load dataset
    loader = DataLoader()
    df = loader.load_data()

    # Validate
    validator = DataValidator(df)
    validator.validate()

    # Clean
    cleaner = DataCleaner(df)
    df = cleaner.clean()

    # Convert
    df = convert_dataset(df)

    print("\n✅ Preprocessing completed successfully.")
    print(f"Final Dataset Shape: {df.shape}")


if __name__ == "__main__":
    main()