"""
=========================================================
SwitchGuardAI - Temporal Features
=========================================================

Builds lag/gradient features - values that compare the
CURRENT operation to the PREVIOUS operation on the SAME switch.

IMPORTANT ON DATA LEAKAGE:
- Using the CURRENT row's Health_Index to predict the CURRENT
  row's Fault_Label is data leakage (cheating).
- Using the PREVIOUS row's Health_Index/Fault to help predict
  the CURRENT row is NOT leakage - it's a standard, legitimate
  technique used in real predictive maintenance research
  (e.g. Atamuradov et al., 2018), since only past information
  is used.

Author : Tanishka
Project : SwitchGuardAI
"""


def build_temporal_features(df):
    """
    Adds gradient (rate of change) and lag features to the
    dataframe.

    IMPORTANT: df must already be sorted by
    ["Switch_ID", "Timestamp"] before calling this.

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain: Switch_ID, Motor_Temperature, Motor_Current,
        Health_Index, Fault_Label

    Returns
    -------
    df : pandas.DataFrame with new columns added
    """

    grouped = df.groupby("Switch_ID")

    # -----------------------------------------------
    # Gradients: how much did this reading change
    # since the last operation on this switch?
    # -----------------------------------------------

    df["Temperature_Gradient"] = (
        grouped["Motor_Temperature"].diff().fillna(0)
    )

    df["Current_Gradient"] = (
        grouped["Motor_Current"].diff().fillna(0)
    )

    # -----------------------------------------------
    # Lag features: the PREVIOUS operation's health/fault
    # state on this switch (not the current row's own value)
    # -----------------------------------------------

    df["Previous_Health"] = (
        grouped["Health_Index"].shift(1)
    )

    # First operation on each switch has no "previous" -
    # assume it started at perfect health (100)
    df["Previous_Health"] = df["Previous_Health"].fillna(100.0)

    df["Previous_Fault"] = (
        grouped["Fault_Label"].shift(1)
    )

    # First operation on each switch has no previous fault -
    # assume it started "Normal"
    df["Previous_Fault"] = df["Previous_Fault"].fillna("Normal")

    return df


# =========================================================
# TEST
# =========================================================

if __name__ == "__main__":

    import pandas as pd

    sample = pd.DataFrame({
        "Switch_ID": ["SW001"] * 4,
        "Motor_Temperature": [40, 45, 50, 60],
        "Motor_Current": [15, 16, 17, 20],
        "Health_Index": [100, 95, 88, 70],
        "Fault_Label": ["Normal", "Normal", "Normal", "Overheat"]
    })

    result = build_temporal_features(sample)

    print(result)