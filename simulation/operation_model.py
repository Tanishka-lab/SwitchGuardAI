"""
=========================================================
SwitchGuardAI Operation Model
=========================================================

This module simulates one railway switch operation.

Outputs:
1. Time gap since previous operation
2. Motor current
3. Operation duration

Physics & Research Inspired:
- Higher traffic -> more frequent operations
- Lower ambient temperature -> slightly higher current
- Faults gradually increase current and duration
- Real mechanical wear builds up with repeated use

Author : Tanishka
Project : SwitchGuardAI
"""

import numpy as np

from configs.sensor_specifications import (
    BASE_CURRENT,
    MAX_CURRENT,
    BASE_OPERATION_DURATION,
    MAX_OPERATION_DURATION,
)


class OperationModel:

    def __init__(self, traffic_level="Medium"):
        self.traffic_level = traffic_level

    def get_operation_gap(self):
        """
        Returns hours between two switch operations.
        """
        if self.traffic_level == "Low":
            return np.random.uniform(6, 24)
        elif self.traffic_level == "Medium":
            return np.random.uniform(2, 12)
        else:
            return np.random.uniform(0.5, 6)

    def simulate_operation(
        self,
        ambient_temperature,
        health_index=100,
        operation_count=0
    ):
        """
        Simulate one switch operation.

        Returns
        -------
        hours_since_last_operation
        motor_current
        operation_duration
        """

        hours_since_last_operation = self.get_operation_gap()

        # Current increases in colder weather
        temperature_effect = max(
            0,
            (25 - ambient_temperature) * 0.08
        )

        # Health degradation effect
        degradation_effect = (100 - health_index) * 0.03

        # Real mechanical wear: builds up slowly just from
        # repeated use, independent of current health.
        # This is what prevents the system from "staying perfect forever"
        wear_effect = operation_count * 0.00025

        # small chance of a sudden obstruction/debris event,
        # which gets more likely as the machine wears
        sudden_fault_chance = 0.004 + (operation_count * 0.0000035)
        sudden_fault_boost = 0.0

        if np.random.rand() < sudden_fault_chance:
            sudden_fault_boost = np.random.uniform(3, 9)

        motor_current = (
            BASE_CURRENT
            + temperature_effect
            + degradation_effect
            + wear_effect
            + sudden_fault_boost
            + np.random.normal(0, 0.35)
        )

        motor_current = np.clip(motor_current, 10, MAX_CURRENT)

        operation_duration = (
            BASE_OPERATION_DURATION
            + degradation_effect * 0.15
            + wear_effect * 0.3
            + sudden_fault_boost * 0.4
            + np.random.normal(0, 0.15)
        )

        operation_duration = np.clip(operation_duration, 3, MAX_OPERATION_DURATION)

        return (
            round(hours_since_last_operation, 2),
            round(float(motor_current), 2),
            round(float(operation_duration), 2)
        )


if __name__ == "__main__":
    model = OperationModel()
    gap, current, duration = model.simulate_operation(
        ambient_temperature=30,
        health_index=95
    )
    print("Hours Gap :", gap)
    print("Motor Current :", current)
    print("Duration :", duration)