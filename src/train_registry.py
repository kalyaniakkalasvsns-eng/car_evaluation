from pathlib import Path
import sys
import subprocess
import mlflow
import mlflow.sklearn


BASE_DIR = Path(__file__).resolve().parents[1]
PYTHON = sys.executable


def main():

    print("[INFO] Training model for Model Registry...")

    # Run preprocessing
    subprocess.run(
        [PYTHON, str(BASE_DIR / "src" / "preprocess.py")],
        check=True
    )

    # Run training
    subprocess.run(
        [PYTHON, str(BASE_DIR / "src" / "train.py")],
        check=True
    )

    # Load trained model
    import joblib

    model = joblib.load(
        BASE_DIR / "models" / "random_forest_model.pkl"
    )

    # MLflow experiment
    mlflow.set_experiment("Car_Evaluation_RandomForest")

    with mlflow.start_run() as run:

        mlflow.log_param(
            "model",
            "RandomForestClassifier"
        )

        mlflow.log_param(
            "n_estimators",
            200
        )

        mlflow.log_param(
            "random_state",
            42
        )

        mlflow.log_param(
            "class_weight",
            "balanced"
        )

        # Register model in MLflow Model Registry
        mlflow.sklearn.log_model(
            model,
            name="car_evaluation_model",
            registered_model_name="CarEvaluationModel",
            skops_trusted_types=[
                "sklearn.tree._tree.Tree"
            ]
        )

        print("\n[SUCCESS] Model registered successfully.")
        print(f"[INFO] Run ID: {run.info.run_id}")
        print("[INFO] Registered Model: CarEvaluationModel")


if __name__ == "__main__":
    main()