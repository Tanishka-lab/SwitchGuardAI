"""
=========================================================
SwitchGuardAI - Rolling Features
=========================================================

Builds moving-average / rolling statistics, computed
PER SWITCH (never mixing one switch's history with another's).

These help the model detect gradual heating/wear trends,
instead of reacting to one single noisy reading.

Author : Tanishka
Project : SwitchGuardAI
"""


def build_rolling_features(df, window=5):
    """
    Adds rolling average features to the dataframe.

    IMPORTANT: df must already be sorted by
    ["Switch_ID", "Timestamp"] before calling this,
    otherwise the rolling windows will mix up switches
    or be out of chronological order.

    Parameters
    ----------
    df : pandas.DataFrame
    window : int
        Number of previous operations to average over

    Returns
    -------
    df : pandas.DataFrame with new columns added
    """

    df["Rolling_Avg_Motor_Temp"] = (
        df.groupby("Switch_ID")["Motor_Temperature"]
        .transform(lambda x: x.rolling(window, min_periods=1).mean())
    )

    df["Rolling_Avg_Current"] = (
        df.groupby("Switch_ID")["Motor_Current"]
        .transform(lambda x: x.rolling(window, min_periods=1).mean())
    )

    df["Rolling_Std_Current"] = (
        df.groupby("Switch_ID")["Motor_Current"]
        .transform(lambda x: x.rolling(window, min_periods=1).std())
        .fillna(0)
    )

    return df


# =========================================================
# TEST
# =========================================================

if __name__ == "__main__":

    import pandas as pd

    sample = pd.DataFrame({
        "Switch_ID": ["SW001"] * 6,
        "Motor_Temperature": [40, 42, 45, 50, 55, 60],
        "Motor_Current": [15, 15.5, 16, 17, 18, 19]
    })

    result = build_rolling_features(sample, window=3)

    print(result)