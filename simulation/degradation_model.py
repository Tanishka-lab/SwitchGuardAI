"""
=========================================================
SwitchGuardAI Degradation Model
=========================================================

Computes the Health Index (0–100) based on:

1. Motor Temperature
2. Motor Current
3. Operation Duration
4. Days Since Maintenance

Author : Tanishka
Project : SwitchGuardAI
"""

from configs.sensor_specifications import (
    MAX_MOTOR_TEMP,
    MAX_CURRENT,
    MAX_OPERATION_DURATION,
    MAINTENANCE_INTERVAL
)


class DegradationModel:

    def calculate(
        self,
        motor_temp,
        motor_current,
        operation_duration,
        maintenance_days
    ):

        health = 100.0

        # -----------------------------------------
        # Temperature degradation
        # -----------------------------------------

        if motor_temp > 50:

            health -= (motor_temp - 50) * 1.2

        # -----------------------------------------
        # Current degradation
        # -----------------------------------------

        if motor_current > 16:

            health -= (motor_current - 16) * 6

        # -----------------------------------------
        # Operation duration degradation
        # -----------------------------------------

        if operation_duration > 5:

            health -= (operation_duration - 5) * 5

        # -----------------------------------------
        # Maintenance degradation
        # -----------------------------------------

        if maintenance_days > 60:

            health -= (maintenance_days - 60) * 0.35

        return round(max(0, min(100, health)), 2)