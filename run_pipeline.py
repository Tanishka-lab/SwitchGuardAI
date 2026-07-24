"""
=========================================================
SwitchGuardAI Pipeline Runner
=========================================================

Runs the complete SwitchGuardAI workflow.

Pipeline
--------
1. Dataset Generation
2. Exploratory Data Analysis
3. Data Preprocessing
4. Feature Engineering
5. Model Training
6. Model Evaluation
7. Explainable AI

Author : Tanishka
Project : SwitchGuardAI
=========================================================
"""

import subprocess
import sys


def run_step(title, command):

    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)

    result = subprocess.run(command)

    if result.returncode != 0:
        print(f"\n❌ {title} Failed")
        sys.exit(1)

    print(f"\n✅ {title} Completed Successfully")


def main():

    print("=" * 70)
    print("           SwitchGuardAI Complete Pipeline")
    print("=" * 70)

    # STEP 1
    run_step(
        "STEP 1 : Dataset Generation",
        ["python", "simulation/data_generator.py"]
    )

    # STEP 2
    run_step(
        "STEP 2 : Exploratory Data Analysis",
        ["python", "analysis/eda.py"]
    )

    # STEP 3
    run_step(
        "STEP 3 : Data Preprocessing",
        ["python", "preprocess_data.py"]
    )

    # STEP 4
    run_step(
        "STEP 4 : Feature Engineering",
        ["python", "feature_pipeline.py"]
    )

    # STEP 5
    run_step(
        "STEP 5 : Model Training",
        ["python", "models/train_model.py"]
    )

    # STEP 6
    run_step(
        "STEP 6 : Model Evaluation",
        ["python", "models/evaluate_model.py"]
    )

    # STEP 7
    run_step(
        "STEP 7 : Explainable AI",
        ["python", "models/explain_model.py"]
    )

    print("\n" + "=" * 70)
    print("🎉 SwitchGuardAI Pipeline Completed Successfully")
    print("=" * 70)

    print("\nGenerated Artifacts:")
    print("• data/raw/")
    print("• data/processed/")
    print("• saved_models/")
    print("• results/")


if __name__ == "__main__":
    main()