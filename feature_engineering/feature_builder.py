"""
=========================================================
SwitchGuardAI - Feature Builder
=========================================================

Builds basic physics-based features from raw sensor readings:
- Thermal rise (excess temperature above ambient)
- Thermal stress (current x temperature interaction)
- Current rate (current / duration)
- Maintenance ratio (normalized 0-1)

Author : Tanishka
Project : SwitchGuardAI
"""


def build_physics_features(df):
    """
    Adds physics-based features to the dataframe.

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain: Motor_Temperature, Gearbox_Temperature,
        Lock_Temperature, Control_Cabinet_Temperature,
        Ambient_Temperature, Motor_Current, Operation_Duration,
        Days_Since_Last_Maintenance

    Returns
    -------
    df : pandas.DataFrame with new columns added
    """

    # -----------------------------------------------
    # Thermal rise features
    # (excess heat above ambient - real methodology
    # used in the Dutch NSE2 point machine study)
    # -----------------------------------------------

    df["Thermal_Rise_Motor"] = (
        df["Motor_Temperature"] - df["Ambient_Temperature"]
    )

    df["Thermal_Rise_Gearbox"] = (
        df["Gearbox_Temperature"] - df["Ambient_Temperature"]
    )

    df["Thermal_Rise_Lock"] = (
        df["Lock_Temperature"] - df["Ambient_Temperature"]
    )

    df["Thermal_Rise_Cabinet"] = (
        df["Control_Cabinet_Temperature"] - df["Ambient_Temperature"]
    )

    # -----------------------------------------------
    # Thermal stress (interaction feature)
    # Higher current AND higher temperature together
    # signals more mechanical/electrical stress
    # -----------------------------------------------

    df["Thermal_Stress"] = (
        df["Motor_Current"] * df["Motor_Temperature"]
    )

    # -----------------------------------------------
    # Current rate (previously called "Current Density" -
    # renamed since true current density means A/mm^2,
    # not current-over-time)
    # -----------------------------------------------

    df["Current_Rate"] = (
        df["Motor_Current"] / df["Operation_Duration"]
    )

    # -----------------------------------------------
    # Maintenance ratio: scales days-since-maintenance
    # to a 0-1 range (0 = just maintained, 1 = due)
    # -----------------------------------------------

    MAINTENANCE_INTERVAL = 90

    df["Maintenance_Ratio"] = (
        df["Days_Since_Last_Maintenance"] / MAINTENANCE_INTERVAL
    ).clip(upper=1.0)

    return df


# =========================================================
# TEST
# =========================================================

if __name__ == "__main__":

    import pandas as pd

    sample = pd.DataFrame({
        "Motor_Temperature": [60],
        "Gearbox_Temperature": [50],
        "Lock_Temperature": [40],
        "Control_Cabinet_Temperature": [35],
        "Ambient_Temperature": [30],
        "Motor_Current": [16],
        "Operation_Duration": [4],
        "Days_Since_Last_Maintenance": [45]
    })

    result = build_physics_features(sample)

    print(result.T)