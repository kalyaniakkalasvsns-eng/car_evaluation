# Car Evaluation MLOps Project

This project applies an MLOps workflow to the Car Evaluation dataset.

## Dataset
- File: `data/raw/car_evaluation.csv`
- 1728 records
- 6 categorical input features
- 1 target column: `class`
- Target classes: `unacc`, `acc`, `good`, `vgood`

## Workflow

Raw data
-> validation
-> preprocessing / one-hot encoding
-> train/test split
-> Random Forest
-> evaluation
-> artifacts

## Run

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Install packages:

```bash
pip install -r requirements.txt
```

Run the complete baseline:

```bash
python pipelines/run_lab3_baseline.py
```

Run the validation pipeline:

```bash
python pipelines/run_lab5_pipeline.py
```
