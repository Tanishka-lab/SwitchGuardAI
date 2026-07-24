"""
=========================================================
SwitchGuardAI Fault Model
=========================================================

Converts Health Index into maintenance status.

Author : Tanishka
Project : SwitchGuardAI
"""


class FaultModel:

    def classify(self, health):

        if health >= 85:

            return "Normal"

        elif health >= 70:

            return "Maintenance_Overdue"

        elif health >= 55:

            return "Mechanical_Resistance"

        elif health >= 40:

            return "Overheat"

        else:

            return "Critical"