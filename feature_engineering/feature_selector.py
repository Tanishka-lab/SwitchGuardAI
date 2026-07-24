"""
=========================================================
SwitchGuardAI - Feature Selector
=========================================================

Analyzes correlation between engineered features and
Health_Index, to help decide which features are actually
useful before moving to model building in Phase 2.

NOTE: Health_Index itself and Fault_Label are never treated
as INPUT features for classification - they are only used
HERE, in this analysis step, to judge how useful each
candidate feature is.

Author : Tanishka
Project : SwitchGuardAI
"""

import pandas as pd


def select_features(df, target_col="Health_Index", exclude=None, top_n=15):
    """
    Ranks numeric features by their absolute correlation with
    the target column.

    Parameters
    ----------
    df : pandas.DataFrame
    target_col : str
        Column to measure correlation against (Health_Index
        for analysis purposes only - never used as a model input)
    exclude : list of str
        Columns to exclude from ranking (e.g. IDs, the target
        itself, Fault_Label, Timestamp)
    top_n : int
        How many top features to return

    Returns
    -------
    ranked : pandas.Series
        Features sorted by absolute correlation strength (descending)
    """

    if exclude is None:
        exclude = []

    numeric_df = df.select_dtypes(include="number")

    candidate_cols = [
        c for c in numeric_df.columns
        if c not in exclude and c != target_col
    ]

    correlations = numeric_df[candidate_cols].corrwith(df[target_col])

    ranked = correlations.reindex(
        correlations.abs().sort_values(ascending=False).index
    )

    print(f"Top {top_n} features most correlated with {target_col}:\n")
    print(ranked.head(top_n).round(3))

    return ranked.head(top_n)


# =========================================================
# TEST (standalone run on the full processed dataset)
# =========================================================

if __name__ == "__main__":

    df = pd.read_csv("data/processed/switch_machine_dataset_features.csv")

    ranked = select_features(
        df,
        target_col="Health_Index",
        exclude=["Operation_Count"]
    )

    ranked.to_csv("data/processed/top_features_ranked.csv")

    print("\nSaved ranked feature list to data/processed/top_features_ranked.csv")