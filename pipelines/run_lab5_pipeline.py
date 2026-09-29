import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(script):
    subprocess.run([sys.executable, str(ROOT / "src" / script)], check=True)


if __name__ == "__main__":
    print("=== VALIDATED MLOPS PIPELINE ===")
    run("validate_data.py")
    run("preprocess.py")
    run("train.py")
    run("evaluate.py")
    run("validate_outputs.py")
    print("=== VALIDATED PIPELINE COMPLETED ===")
