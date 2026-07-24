"""
=========================================================
SwitchGuardAI Thermal Model
=========================================================

Simulates the thermal behaviour of a railway switch
machine using:

1. Joule Heating (I^2R)
2. Newton's Law of Cooling

Outputs:
- Motor Temperature
- Gearbox Temperature
- Lock Temperature
- Control Cabinet Temperature

Author : Tanishka
Project : SwitchGuardAI
"""

import numpy as np

from configs.sensor_specifications import (
    MOTOR_RESISTANCE,
    HEAT_SCALING_FACTOR,
    MAX_HEAT_GAIN_PER_OPERATION,
    MOTOR_COOLING_RATE,
    GEARBOX_COOLING_RATE,
    LOCK_COOLING_RATE,
    GEARBOX_HEAT_FACTOR,
    LOCK_HEAT_FACTOR,
    MOTOR_SENSOR_STD,
    GEARBOX_SENSOR_STD,
    LOCK_SENSOR_STD,
    CABINET_SENSOR_STD,
    MAX_MOTOR_TEMP,
    MAX_GEARBOX_TEMP,
    MAX_LOCK_TEMP,
    MAX_CABINET_TEMP
)


class ThermalModel:

    def __init__(self):

        self.motor_temp = 25.0
        self.gearbox_temp = 25.0
        self.lock_temp = 25.0

    # =====================================================

    def update(

        self,

        ambient_temp,

        motor_current,

        operation_duration,

        hours_since_last_operation

    ):

        # =================================================
        # STEP 1 : COOLING
        # =================================================

        self.motor_temp = (

            ambient_temp

            + (self.motor_temp - ambient_temp)

            * np.exp(

                -MOTOR_COOLING_RATE

                * hours_since_last_operation

            )

        )

        self.gearbox_temp = (

            ambient_temp

            + (self.gearbox_temp - ambient_temp)

            * np.exp(

                -GEARBOX_COOLING_RATE

                * hours_since_last_operation

            )

        )

        self.lock_temp = (

            ambient_temp

            + (self.lock_temp - ambient_temp)

            * np.exp(

                -LOCK_COOLING_RATE

                * hours_since_last_operation

            )

        )

        # =================================================
        # STEP 2 : HEAT GENERATED
        # =================================================

        heat_gain = (

            (motor_current ** 2)

            * MOTOR_RESISTANCE

            * operation_duration

            * HEAT_SCALING_FACTOR

        )

        # Prevent unrealistic spikes, but allow real fault severity to show
        heat_gain = min(heat_gain, MAX_HEAT_GAIN_PER_OPERATION)

        # =================================================
        # STEP 3 : HEATING
        # =================================================

        self.motor_temp += heat_gain

        self.gearbox_temp += (

            heat_gain

            * GEARBOX_HEAT_FACTOR

        )

        self.lock_temp += (

            heat_gain

            * LOCK_HEAT_FACTOR

        )

        # =================================================
        # STEP 4 : SENSOR NOISE
        # =================================================

        self.motor_temp += np.random.normal(

            0,

            MOTOR_SENSOR_STD

        )

        self.gearbox_temp += np.random.normal(

            0,

            GEARBOX_SENSOR_STD

        )

        self.lock_temp += np.random.normal(

            0,

            LOCK_SENSOR_STD

        )

        # =================================================
        # STEP 5 : PHYSICAL LIMITS
        # =================================================

        self.motor_temp = np.clip(

            self.motor_temp,

            ambient_temp,

            MAX_MOTOR_TEMP

        )

        self.gearbox_temp = np.clip(

            self.gearbox_temp,

            ambient_temp,

            MAX_GEARBOX_TEMP

        )

        self.lock_temp = np.clip(

            self.lock_temp,

            ambient_temp,

            MAX_LOCK_TEMP

        )

        # =================================================
        # STEP 6 : CONTROL CABINET
        # =================================================

        cabinet_temp = (

            ambient_temp

            + 4

            + 0.10 * motor_current

            + np.random.normal(

                0,

                CABINET_SENSOR_STD

            )

        )

        cabinet_temp = np.clip(

            cabinet_temp,

            ambient_temp,

            MAX_CABINET_TEMP

        )

        return (

            round(float(self.motor_temp), 2),

            round(float(self.gearbox_temp), 2),

            round(float(self.lock_temp), 2),

            round(float(cabinet_temp), 2)

        )


# =========================================================
# TEST
# =========================================================

if __name__ == "__main__":

    model = ThermalModel()

    for _ in range(5):

        values = model.update(

            ambient_temp=32,

            motor_current=14,

            operation_duration=4,

            hours_since_last_operation=3

        )

        print(values)
