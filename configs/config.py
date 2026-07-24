"""
=========================================================
SwitchGuardAI Configuration
=========================================================

Central configuration file for the project.

Author : Tanishka
Project : SwitchGuardAI
=========================================================
"""

from pathlib import Path


# =========================================================
# PROJECT ROOT
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent


# =========================================================
# DATA DIRECTORIES
# =========================================================

DATA_DIR = PROJECT_ROOT / "data"

RAW_DATA_DIR = DATA_DIR / "raw"

PROCESSED_DATA_DIR = DATA_DIR / "processed"


# =========================================================
# DATA FILES
# =========================================================

RAW_DATASET = RAW_DATA_DIR / "switch_machine_dataset.csv"

CLEAN_DATASET = PROCESSED_DATA_DIR / "switch_machine_dataset_clean.csv"

FEATURE_DATASET = PROCESSED_DATA_DIR / "switch_machine_dataset_features.csv"

TRAIN_DATASET = PROCESSED_DATA_DIR / "train_dataset.csv"

TEST_DATASET = PROCESSED_DATA_DIR / "test_dataset.csv"

X_TRAIN = PROCESSED_DATA_DIR / "X_train.csv"

Y_TRAIN = PROCESSED_DATA_DIR / "y_train.csv"

X_TEST = PROCESSED_DATA_DIR / "X_test.csv"

Y_TEST = PROCESSED_DATA_DIR / "y_test.csv"


# =========================================================
# RESULTS
# =========================================================

RESULTS_DIR = PROJECT_ROOT / "results"


# =========================================================
# REPORTS
# =========================================================

REPORTS_DIR = PROJECT_ROOT / "reports"


# =========================================================
# SAVED MODELS
# =========================================================

MODELS_DIR = PROJECT_ROOT / "saved_models"

RANDOM_FOREST_MODEL = MODELS_DIR / "random_forest.pkl"

FEATURE_NAMES = MODELS_DIR / "feature_names.pkl"


# =========================================================
# FEATURE ENGINEERING
# =========================================================

ROLLING_WINDOW = 5


# =========================================================
# TRAINING
# =========================================================

TEST_SIZE = 0.20

RANDOM_STATE = 42

N_ESTIMATORS = 200


# =========================================================
# DIRECTORIES TO CREATE
# =========================================================

DIRECTORIES = [

    RAW_DATA_DIR,

    PROCESSED_DATA_DIR,

    RESULTS_DIR,

    REPORTS_DIR,

    MODELS_DIR

]