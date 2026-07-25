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
import os

# Always run commands from the project root
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

def run_step(title, command):

    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)

    result = subprocess.run(command, cwd=PROJECT_ROOT)

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
        [sys.executable, "-m" ,"simulation.data_generator"]
    )

    # STEP 2
    run_step(
        "STEP 2 : Exploratory Data Analysis",
        [sys.executable, "-m" ,"analysis.eda"]
    )

    # STEP 3
    run_step(
        "STEP 3 : Data Preprocessing",
        [sys.executable, "preprocess_data.py"]
    )

    # STEP 4
    run_step(
        "STEP 4 : Feature Engineering",
        [sys.executable, "feature_pipeline.py"]
    )

    # STEP 5
    run_step(
        "STEP 5 : Model Training",
        [sys.executable,"-m" , "models.train_model"]
    )

    # STEP 6
    run_step(
        "STEP 6 : Model Evaluation",
        [sys.executable,"-m" , "models.evaluate_model"]
    )

    # STEP 7
    run_step(
        "STEP 7 : Explainable AI",
        [sys.executable,"-m" , "models.explain_model"]
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