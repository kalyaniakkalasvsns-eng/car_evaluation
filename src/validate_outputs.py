from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"
OUT_DIR = ROOT / "outputs"
ARTIFACT_DIR = ROOT / "artifacts"


def main():
    required = [
        DATA_DIR / "X_train.npy",
        DATA_DIR / "X_test.npy",
        DATA_DIR / "y_train.npy",
        DATA_DIR / "y_test.npy",
        DATA_DIR / "dataset_metadata.json",
        MODEL_DIR / "encoder.pkl",
        MODEL_DIR / "random_forest_model.pkl",
        OUT_DIR / "classification_report.json",
        OUT_DIR / "confusion_matrix.png",
    ]

    missing = [str(p) for p in required if not p.exists()]

    report = {
        "valid": len(missing) == 0,
        "missing_files": missing
    }

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    with open(ARTIFACT_DIR / "output_validation_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    if missing:
        raise FileNotFoundError("Missing output files:\n" + "\n".join(missing))

    print("Output validation PASSED.")


if __name__ == "__main__":
    main()
