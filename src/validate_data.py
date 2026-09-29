from pathlib import Path
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_FILE = ROOT / "data" / "raw" / "car_evaluation.csv"
ARTIFACT_DIR = ROOT / "artifacts"

COLUMNS = ["buying", "maint", "doors", "persons", "lug_boot", "safety", "class"]

EXPECTED = {
    "buying": {"vhigh", "high", "med", "low"},
    "maint": {"vhigh", "high", "med", "low"},
    "doors": {"2", "3", "4", "5more"},
    "persons": {"2", "4", "more"},
    "lug_boot": {"small", "med", "big"},
    "safety": {"low", "med", "high"},
    "class": {"unacc", "acc", "good", "vgood"}
}


def main():
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(RAW_FILE, header=None, names=COLUMNS)

    errors = []

    if list(df.columns) != COLUMNS:
        errors.append("Column names are incorrect.")

    if df.isnull().any().any():
        errors.append("Missing values found.")

    for col, allowed in EXPECTED.items():
        actual = set(df[col].astype(str).unique())
        unknown = actual - allowed
        if unknown:
            errors.append(f"{col}: unexpected values {sorted(unknown)}")

    report = {
        "valid": len(errors) == 0,
        "rows": int(len(df)),
        "columns": COLUMNS,
        "errors": errors
    }

    with open(ARTIFACT_DIR / "validation_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    if errors:
        raise ValueError("\n".join(errors))

    print("Data validation PASSED.")
    print(f"Rows: {len(df)}")


if __name__ == "__main__":
    main()
