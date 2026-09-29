from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parents[1]
RAW_FILE = ROOT / "data" / "raw" / "car_evaluation.csv"
OUT_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"

COLUMNS = ["buying", "maint", "doors", "persons", "lug_boot", "safety", "class"]
FEATURES = COLUMNS[:-1]
TARGET = "class"


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(RAW_FILE, header=None, names=COLUMNS)

    if df.isnull().any().any():
        raise ValueError("Dataset contains missing values.")

    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    X_train_encoded = encoder.fit_transform(X_train)
    X_test_encoded = encoder.transform(X_test)

    np.save(OUT_DIR / "X_train.npy", X_train_encoded)
    np.save(OUT_DIR / "X_test.npy", X_test_encoded)
    np.save(OUT_DIR / "y_train.npy", y_train.to_numpy())
    np.save(OUT_DIR / "y_test.npy", y_test.to_numpy())

    import joblib
    joblib.dump(encoder, MODEL_DIR / "encoder.pkl")

    metadata = {
        "rows": int(len(df)),
        "features": FEATURES,
        "target": TARGET,
        "classes": sorted(y.unique().tolist()),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "encoded_features": int(X_train_encoded.shape[1]),
        "random_state": 42
    }

    with open(OUT_DIR / "dataset_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("Preprocessing completed.")
    print(f"Train rows: {len(X_train)}")
    print(f"Test rows : {len(X_test)}")
    print(f"Encoded features: {X_train_encoded.shape[1]}")


if __name__ == "__main__":
    main()
