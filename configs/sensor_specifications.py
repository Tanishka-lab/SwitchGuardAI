"""
=========================================================
SwitchGuardAI - Sensor Specifications (Config File)
=========================================================

All constants used across the simulation modules live here,
in ONE place, so they are easy to find and change.

Values are based on:
- I2R heating physics
- IEC 60034 motor thermal limits
- Real published research (Cu-3300, Network Rail UK,
  Dutch NSE2 study)
=========================================================
"""

# ---------------------------------------------------------
# THERMAL MODEL CONSTANTS
# ---------------------------------------------------------

MOTOR_RESISTANCE = 1.8          # ohms (effective resistance constant)
HEAT_SCALING_FACTOR = 0.0025      # scales I^2 * R * t down to realistic degrees
MAX_HEAT_GAIN_PER_OPERATION = 35

MOTOR_COOLING_RATE = 0.5        # how fast motor cools toward ambient
GEARBOX_COOLING_RATE = 0.2      # gearbox has more mass -> cools slower
LOCK_COOLING_RATE = 0.8         # lock is smaller -> cools faster

GEARBOX_HEAT_FACTOR = 0.6       # gearbox heats up less than the motor itself
LOCK_HEAT_FACTOR = 0.4          # lock heats up even less

MOTOR_SENSOR_STD = 1.0          # sensor noise (standard deviation, in deg C)
GEARBOX_SENSOR_STD = 0.8
LOCK_SENSOR_STD = 0.6
CABINET_SENSOR_STD = 0.5

# Max safe temperatures (deg C) - based on IEC 60034 Class B/F limits,
# scaled down for smaller components like lock/cabinet
MAX_MOTOR_TEMP = 140
MAX_GEARBOX_TEMP = 110
MAX_LOCK_TEMP = 100
MAX_CABINET_TEMP = 70

# ---------------------------------------------------------
# OPERATION MODEL CONSTANTS
# ---------------------------------------------------------

BASE_CURRENT = 14.0              # Amps, normal healthy operation
MAX_CURRENT = 30.0                # Amps, absolute ceiling (fault conditions)

BASE_OPERATION_DURATION = 4.0     # seconds, normal healthy throw
MAX_OPERATION_DURATION = 15.0      # seconds, absolute ceiling (fault conditions)

# ---------------------------------------------------------
# DEGRADATION MODEL CONSTANTS
# ---------------------------------------------------------

MAINTENANCE_INTERVAL = 90         # days, typical maintenance cycle

# ---------------------------------------------------------
# LOCATION TYPES (affects how often the switch is used)
# ---------------------------------------------------------

SWITCH_LOCATIONS = {

    "SW001": "Junction",
    "SW002": "Junction",

    "SW003": "Suburban",
    "SW004": "Suburban",
    "SW005": "Suburban",

    "SW006": "Rural",
    "SW007": "Rural",
    "SW008": "Rural",
    "SW009": "Rural",
    "SW010": "Rural"

}