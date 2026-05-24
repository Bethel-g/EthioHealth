"""
Quick Start Script for EthioHealth-AI
Runs the complete pipeline: data generation -> preprocessing -> training -> API
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(cmd, description):
    """Run a command and report status."""
    print(f"\n{'='*70}")
    print(f"⏳ {description}")
    print(f"{'='*70}")
    try:
        result = subprocess.run(cmd, shell=True, check=True)
        print(f"✓ {description} - COMPLETE")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {description} - FAILED")
        print(f"Error: {e}")
        return False


def main():
    """Run complete pipeline."""
    print("""
    ╔════════════════════════════════════════════════════════════════╗
    ║                   EthioHealth-AI Quick Start                   ║
    ║         Localized Clinical Decision Support System             ║
    ╚════════════════════════════════════════════════════════════════╝
    """)

    # Ensure we're in correct directory
    project_root = Path(__file__).parent
    os.chdir(project_root)

    # Step 1: Ensure directories exist
    print("\n📁 Creating directories...")
    for directory in ["data", "models", "notebooks/figures"]:
        Path(directory).mkdir(parents=True, exist_ok=True)
    print("✓ Directories ready")

    # Step 2: Data Generation
    if not run_command(
        f"{sys.executable} src/data_generator.py",
        "Phase 1: Generating synthetic Ethiopian ED data"
    ):
        print("⚠️  Continuing despite data generation issues...")

    # Step 3: Preprocessing
    if not run_command(
        f"{sys.executable} src/preprocessing.py",
        "Phase 2: Preprocessing data"
    ):
        print("⚠️  Continuing despite preprocessing issues...")

    # Step 4: EDA
    if not run_command(
        f"{sys.executable} src/eda.py",
        "Phase 3: Exploratory Data Analysis"
    ):
        print("⚠️  Continuing despite EDA issues...")

    # Step 5: Model Training
    if not run_command(
        f"{sys.executable} src/model_training.py",
        "Phase 4: Training regression models"
    ):
        print("✗ Model training failed. Cannot continue.")
        sys.exit(1)

    # Complete
    print(f"\n{'='*70}")
    print("✓ Pipeline Complete!")
    print(f"{'='*70}")

    print("""
    🎉 EthioHealth-AI is ready for deployment!

    Next Steps:
    
    1. START API SERVER (Terminal 1):
       python -m uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
    
    2. START STREAMLIT DASHBOARD (Terminal 2):
       streamlit run dashboard/app.py
    
    3. OPEN IN BROWSER:
       - Dashboard: http://localhost:8501
       - API Docs: http://localhost:8000/docs
    
    📊 Available Outputs:
    - Preprocessed data: data/X_preprocessed.csv, data/y_preprocessed.csv
    - Trained models: models/*.pkl
    - EDA plots: notebooks/figures/*.png

    📚 Read the comprehensive README for more details!
    """)


if __name__ == "__main__":
    main()
