"""
=========================================================
SwitchGuardAI - Data Splitting
=========================================================

Splits the dataset into TRAIN and TEST sets correctly for
time-series-per-switch data:

- For EACH switch, the first 80% of its operations (by time)
  go to TRAIN, and the last 20% go to TEST.
- This mimics real deployment: "learn from the past,
  predict the future" - and avoids leaking near-duplicate
  rolling/lag feature rows across train and test.

We do NOT use a random row shuffle split here, because
rolling averages and lag features make consecutive rows
highly related - a random split would leak information
and make results look artificially better than they are.

Author : Tanishka
Project : SwitchGuardAI
"""

import pandas as pd


# Columns that must NEVER be used as model input features
# (IDs, timestamps, the target itself, and Health_Index which
# would leak the answer directly into Fault_Label prediction)
NON_FEATURE_COLUMNS = [
    "Timestamp",
    "Switch_ID",
    "Health_Index",
    "Fault_Label"
]


def split_train_test(df, test_fraction=0.2):
    """
    Splits the dataframe into train/test sets, per switch,
    preserving chronological order.

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain Switch_ID and Timestamp columns
    test_fraction : float
        Fraction of EACH switch's operations to hold out
        for testing (taken from the END of its timeline)

    Returns
    -------
    train_df, test_df : pandas.DataFrame, pandas.DataFrame
    """

    df = df.copy()
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    df = df.sort_values(["Switch_ID", "Timestamp"]).reset_index(drop=True)

    train_parts = []
    test_parts = []

    for switch_id, group in df.groupby("Switch_ID"):

        split_point = int(len(group) * (1 - test_fraction))

        train_parts.append(group.iloc[:split_point])
        test_parts.append(group.iloc[split_point:])

    train_df = pd.concat(train_parts).reset_index(drop=True)
    test_df = pd.concat(test_parts).reset_index(drop=True)

    return train_df, test_df


def get_features_and_target(df, target_col="Fault_Label"):
    """
    Separates a dataframe into X (features) and y (target),
    dropping all non-feature columns.

    Returns
    -------
    X : pandas.DataFrame (features only)
    y : pandas.Series (target labels)
    """

    drop_cols = [c for c in NON_FEATURE_COLUMNS if c in df.columns]

    X = df.drop(columns=drop_cols)
    y = df[target_col]

    return X, y


# =========================================================
# TEST
# =========================================================

if __name__ == "__main__":

    df = pd.read_csv("data/processed/switch_machine_dataset_features.csv")

    train_df, test_df = split_train_test(df, test_fraction=0.2)
    import os

    os.makedirs("data/processed", exist_ok=True)

    train_df.to_csv(
    "data/processed/train_dataset.csv",
    index=False
    )

    test_df.to_csv(
    "data/processed/test_dataset.csv",
    index=False
    )

    print("Full dataset shape :", df.shape)
    print("Train shape        :", train_df.shape)
    print(
    f"\nTrain Percentage : {len(train_df)/len(df):.1%}"
    )
    print(
    f"Test Percentage : {len(test_df)/len(df):.1%}"
    )
    
    print("Test shape         :", test_df.shape)

    print("\nFault distribution in TRAIN:")
    print(train_df["Fault_Label"].value_counts(normalize=True).round(4))

    print("\nFault distribution in TEST:")
    print(test_df["Fault_Label"].value_counts(normalize=True).round(4))

    X_train, y_train = get_features_and_target(train_df)
    X_test, y_test = get_features_and_target(test_df)

    print("\nX_train shape:", X_train.shape)
    print("Feature columns used:")
    print(X_train.columns.tolist())