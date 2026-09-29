from pathlib import Path
import sys
import subprocess
import mlflow
import mlflow.sklearn
import numpy as np
import joblib

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

BASE_DIR = Path(__file__).resolve().parents[1]
PYTHON = sys.executable


def main():

    print("=== LAB 4 MLflow EXPERIMENT TRACKING ===")

    # Run existing preprocessing script
    subprocess.run(
        [PYTHON, str(BASE_DIR / "src" / "preprocess.py")],
        check=True
    )

    # Run existing training script
    subprocess.run(
        [PYTHON, str(BASE_DIR / "src" / "train.py")],
        check=True
    )

    # Load trained model and test data
    model = joblib.load(
        BASE_DIR / "models" / "random_forest_model.pkl"
    )

    X_test = np.load(
        BASE_DIR / "data" / "processed" / "X_test.npy"
    )

    y_test = np.load(
        BASE_DIR / "data" / "processed" / "y_test.npy",
        allow_pickle=True
    )

    # Predictions
    y_pred = model.predict(X_test)

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(
        y_test, y_pred,
        average="macro",
        zero_division=0
    )

    recall = recall_score(
        y_test, y_pred,
        average="macro",
        zero_division=0
    )

    f1 = f1_score(
        y_test, y_pred,
        average="macro",
        zero_division=0
    )

    # MLflow experiment
    mlflow.set_experiment("Car_Evaluation_RandomForest")

    with mlflow.start_run() as run:

        # Parameters
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

        # Metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)

        # Model
        mlflow.sklearn.log_model(
    model,
    name="random_forest_model",
    skops_trusted_types=["sklearn.tree._tree.Tree"]
)

        print("\nMLflow Tracking Completed.")
        print(f"Accuracy : {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall   : {recall:.4f}")
        print(f"F1-score : {f1:.4f}")
        print(f"Run ID   : {run.info.run_id}")

    print("\n=== LAB 4 COMPLETED ===")


if __name__ == "__main__":
    main()