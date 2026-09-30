# 241FA18473 - MLOps Experiments 1 to 6 (Kaggle Loan Prediction)

Dataset: Kaggle Loan Prediction Problem Dataset by altruistdelhite04.
Expected files: train_u6lujuX_CVtuZ9i.csv and test_Y3wMUE5_7gLdaTN.csv.

## Dataset placement
No renaming is required. Put the two Kaggle CSV files in `data/raw/`.
The code also automatically finds them if they are inside a folder such as `archive (5)` anywhere under the project.

## VS Code run commands
Open this `241fa18473` folder in VS Code, then Terminal > New Terminal.

Recommended isolated environment:
```
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Run experiments:
```
python pipelines\run_lab3_baseline.py
python pipelines\run_lab4_tracking.py
python pipelines\run_lab5_pipeline.py
python pipelines\run_lab6_registry.py
```

MLflow UI after Lab 4 or 6:
```
python -m mlflow ui --backend-store-uri sqlite:///mlflow.db
```
Open http://127.0.0.1:5000

Optional Kaggle test predictions after training:
```
python src\predict_kaggle_test.py
```
Output: `outputs/kaggle_test_predictions.csv`.

## Correct dataset schema
Target is `Loan_Status` (Y/N). `Loan_ID` is treated only as an identifier and is excluded from model training. Missing numeric values are median-imputed; missing categorical values are most-frequent-imputed; categoricals are one-hot encoded and numeric features are standardized.
