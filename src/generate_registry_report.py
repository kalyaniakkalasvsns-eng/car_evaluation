from pathlib import Path
import json
from mlflow.tracking import MlflowClient


BASE_DIR = Path(__file__).resolve().parents[1]

MODEL_NAME = "CarEvaluationModel"


def main():

    print("[INFO] Generating Model Registry report...")

    client = MlflowClient()

    versions = client.search_model_versions(
        f"name='{MODEL_NAME}'"
    )

    if not versions:
        print("[ERROR] No registered model versions found.")
        return

    latest_version = max(
        versions,
        key=lambda v: int(v.version)
    )

    report = {
        "model_name": MODEL_NAME,
        "latest_version": int(latest_version.version),
        "run_id": latest_version.run_id,
        "source": latest_version.source,
        "status": "Champion"
    }

    artifacts_dir = BASE_DIR / "artifacts"
    artifacts_dir.mkdir(exist_ok=True)

    report_file = artifacts_dir / "registry_report.json"

    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("[SUCCESS] Registry report generated.")
    print(f"[INFO] Report saved at: {report_file}")


if __name__ == "__main__":
    main()