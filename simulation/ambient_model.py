"""
=========================================================
SwitchGuardAI Ambient Environment Model
=========================================================

Generates realistic ambient temperatures for railway
switch machines based on:

1. Season
2. Time of day
3. Weather randomness

Author : Tanishka
Project : SwitchGuardAI
"""

import numpy as np
from datetime import datetime


# =========================================================
# DETERMINE SEASON
# =========================================================

def get_season(date: datetime):

    month = date.month

    if month in [12, 1, 2]:
        return "Winter"

    elif month in [3, 4, 5, 6]:
        return "Summer"

    else:
        return "Monsoon"


# =========================================================
# AMBIENT TEMPERATURE MODEL
# =========================================================

def get_ambient_temperature(date: datetime):
    """
    Generates realistic ambient temperature for India.

    Returns
    -------
    season : str
    ambient_temperature : float
    """

    season = get_season(date)

    hour = date.hour

    # -------------------------------------------------
    # Seasonal average temperatures
    # -------------------------------------------------

    if season == "Winter":
        base_temperature = 18

    elif season == "Summer":
        base_temperature = 36

    else:
        base_temperature = 28

    # -------------------------------------------------
    # Daily temperature cycle
    #
    # Lowest : around 5-6 AM
    # Highest: around 2-3 PM
    # -------------------------------------------------

    daily_variation = 5 * np.sin(
        ((hour - 8) / 24) * 2 * np.pi
    )

    # -------------------------------------------------
    # Random weather variation
    # -------------------------------------------------

    weather_noise = np.random.normal(
        loc=0,
        scale=1
    )

    ambient_temperature = (

        base_temperature

        + daily_variation

        + weather_noise

    )

    # -------------------------------------------------
    # Keep values realistic
    # -------------------------------------------------

    ambient_temperature = np.clip(

        ambient_temperature,

        12,

        45

    )

    return (

        season,

        round(float(ambient_temperature), 2)

    )


# =========================================================
# TEST
# =========================================================

if __name__ == "__main__":

    now = datetime.now()

    season, temperature = get_ambient_temperature(now)

    print("Season :", season)

    print("Ambient Temperature :", temperature)