"""
=========================================================
SwitchGuardAI Dataset Generator
=========================================================

Generates a realistic railway switch machine dataset by
combining:

1. Ambient Model
2. Operation Model
3. Thermal Model
4. Degradation Model
5. Fault Model

Author : Tanishka
Project : SwitchGuardAI
"""

import os
import random
import pandas as pd

from datetime import datetime, timedelta

from simulation.ambient_model import get_ambient_temperature
from simulation.operation_model import OperationModel
from simulation.thermal_model import ThermalModel
from simulation.degradation_model import DegradationModel
from simulation.fault_model import FaultModel

from configs.sensor_specifications import SWITCH_LOCATIONS

# =====================================================
# SETTINGS
# =====================================================

NUMBER_OF_SWITCHES = 10
OPERATIONS_PER_SWITCH = 10000
START_DATE = datetime(2024, 1, 1)
OUTPUT_PATH = "data/raw/switch_machine_dataset.csv"


class SwitchMachineDatasetGenerator:

    def __init__(self):
        self.degradation_model = DegradationModel()
        self.fault_model = FaultModel()
        self.rows = []

    def generate(self):
        print("=" * 60)
        print("SwitchGuardAI Synthetic Dataset Generator")
        print("=" * 60)

        for switch in range(1, NUMBER_OF_SWITCHES + 1):
            switch_id = f"SW{switch:03d}"
            print(f"Generating {switch_id}...")
            self.generate_switch_data(switch_id)

        dataset = pd.DataFrame(self.rows)

        # make sure the output folder exists before saving
        os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
        dataset.to_csv(OUTPUT_PATH, index=False)

        return dataset

    def generate_switch_data(self, switch_id):

        thermal_model = ThermalModel()
        current_time = START_DATE
        operation_count = 0
        days_since_maintenance = 0

        location = SWITCH_LOCATIONS[switch_id]

        if location == "Rural":
            operation_model = OperationModel("Low")
        elif location == "Suburban":
            operation_model = OperationModel("Medium")
        else:
            operation_model = OperationModel("High")

        health = 100

        for _ in range(OPERATIONS_PER_SWITCH):

            # STEP 1: how long since the last operation
            hours_gap = operation_model.get_operation_gap()

            # advance the clock FIRST...
            current_time += timedelta(hours=hours_gap)

            # ...THEN read ambient temperature at this new time
            # (fixes the timestamp/ambient mismatch from the original version)
            season, ambient = get_ambient_temperature(current_time)

            # STEP 2: simulate current & duration for this throw
            (
                _,
                motor_current,
                operation_duration
            ) = operation_model.simulate_operation(
                ambient,
                health,
                operation_count
            )

            operation_count += 1
            days_since_maintenance += hours_gap / 24

            if days_since_maintenance >= 90:
                days_since_maintenance = 0

            # STEP 3: update temperatures
            (
                motor_temp,
                gearbox_temp,
                lock_temp,
                cabinet_temp
            ) = thermal_model.update(
                ambient,
                motor_current,
                operation_duration,
                hours_gap
            )

            # STEP 4: health index
            health = self.degradation_model.calculate(
                motor_temp,
                motor_current,
                operation_duration,
                days_since_maintenance
            )

            # STEP 5: fault classification
            fault = self.fault_model.classify(health)

            # STEP 6: save row
            self.rows.append({
                "Timestamp": current_time,
                "Switch_ID": switch_id,
                "Location_Type": location,
                "Season": season,
                "Ambient_Temperature": ambient,
                "Motor_Temperature": motor_temp,
                "Gearbox_Temperature": gearbox_temp,
                "Lock_Temperature": lock_temp,
                "Control_Cabinet_Temperature": cabinet_temp,
                "Motor_Current": motor_current,
                "Operation_Count": operation_count,
                "Operation_Duration": operation_duration,
                "Days_Since_Last_Maintenance": round(days_since_maintenance, 2),
                "Health_Index": health,
                "Fault_Label": fault
            })


if __name__ == "__main__":
    generator = SwitchMachineDatasetGenerator()
    dataset = generator.generate()

    print("\nDataset Generated Successfully!")
    print("\nRows :", len(dataset))
    print("\nColumns :")
    print(dataset.columns.tolist())
    print("\nFault Distribution")
    print(dataset["Fault_Label"].value_counts())
    print("\nFirst Five Rows")
    print(dataset.head())