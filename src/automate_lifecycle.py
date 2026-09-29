from pathlib import Path
import json
import mlflow
from mlflow.tracking import MlflowClient


BASE_DIR = Path(__file__).resolve().parents[1]

MODEL_NAME = "CarEvaluationModel"


def main():

    print("[INFO] Starting model lifecycle automation...")

    client = MlflowClient()

    # Get all registered model versions
    versions = client.search_model_versions(
        f"name='{MODEL_NAME}'"
    )

    if not versions:
        print("[ERROR] No registered model versions found.")
        return

    print(f"[INFO] Found {len(versions)} model version(s).")

    # Select latest version
    latest_version = max(
        versions,
        key=lambda v: int(v.version)
    )

    version_number = latest_version.version

    print(
        f"[INFO] Latest model version: {version_number}"
    )

    # Promote latest version to Champion
    try:
        client.set_model_version_tag(
            name=MODEL_NAME,
            version=version_number,
            key="lifecycle_stage",
            value="Champion"
        )
    except Exception as e:
        print(f"[WARNING] Could not set tag: {e}")

    # Save lifecycle information
    report = {
        "model_name": MODEL_NAME,
        "champion_version": int(version_number),
        "lifecycle_stage": "Champion",
        "run_id": latest_version.run_id
    }

    output_dir = BASE_DIR / "artifacts"
    output_dir.mkdir(exist_ok=True)

    output_file = output_dir / "model_lifecycle.json"

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("[SUCCESS] Champion model promoted.")
    print(f"[INFO] Champion version: {version_number}")
    print(f"[INFO] Lifecycle report: {output_file}")


if __name__ == "__main__":
    main()