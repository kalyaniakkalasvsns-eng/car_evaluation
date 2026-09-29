from pathlib import Path
import json
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"
OUT_DIR = ROOT / "outputs"


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    X_test = np.load(DATA_DIR / "X_test.npy")
    y_test = np.load(DATA_DIR / "y_test.npy", allow_pickle=True)

    model = joblib.load(MODEL_DIR / "random_forest_model.pkl")
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(
        y_test, y_pred, output_dict=True, zero_division=0
    )

    results = {
        "accuracy": float(accuracy),
        "macro_precision": float(report["macro avg"]["precision"]),
        "macro_recall": float(report["macro avg"]["recall"]),
        "macro_f1": float(report["macro avg"]["f1-score"])
    }

    with open(OUT_DIR / "classification_report.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
    disp.plot()
    plt.title("Car Evaluation - Confusion Matrix")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "confusion_matrix.png")
    plt.close()

    misclassified = pd.DataFrame({
        "actual": y_test,
        "predicted": y_pred
    })
    misclassified = misclassified[misclassified["actual"] != misclassified["predicted"]]
    misclassified.to_csv(OUT_DIR / "misclassified.csv", index=False)

    print("Evaluation completed.")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro F1: {results['macro_f1']:.4f}")


if __name__ == "__main__":
    main()
