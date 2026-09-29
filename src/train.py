from pathlib import Path
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    X_train = np.load(DATA_DIR / "X_train.npy")
    y_train = np.load(DATA_DIR / "y_train.npy", allow_pickle=True)

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced"
    )
    model.fit(X_train, y_train)

    joblib.dump(model, MODEL_DIR / "random_forest_model.pkl")

    print("Model training completed.")
    print("Model: RandomForestClassifier")
    print("Saved: models/random_forest_model.pkl")


if __name__ == "__main__":
    main()
