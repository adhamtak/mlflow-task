# ML Experiment Tracking with MLflow

## Description
This project predicts California house prices with an XGBoost regressor and uses MLflow
to track three experiments with different hyperparameters. For each run, MLflow records
the parameters (`max_depth`, `learning_rate`), the validation metrics (RMSE, MAE, R²),
and the trained model as an artifact. The best model is selected by lowest RMSE.

- Dataset: California Housing (scikit-learn)
- Split: 80% training / 20% validation, `random_state=42`
- Model: XGBoost, 100 boosting rounds

## How to Run

```powershell
python -m venv mlops-env
mlops-env\Scripts\Activate.ps1
pip install -r requirements.txt
```

Terminal 1 (leave running):

```powershell
mlflow server --backend-store-uri sqlite:///mlruns.db --default-artifact-root ./artifacts --host 127.0.0.1 --port 5000
```

Terminal 2 (same folder, venv activated):

```powershell
python train.py
```

Then open http://127.0.0.1:5000 and select the `California_Housing_Optimization` experiment.

## MLflow Experiment Results

![MLflow UI](screenshots/mlflow_runs.png)

## Comparison Table

| Run   | Max Depth | Learning Rate | RMSE | MAE | R² |
|-------|-----------|---------------|------|-----|----|
| Run 1 | 3         | 0.1           | 0.539|0.368|0.779|
| Run 2 | 5         | 0.05          | 0.522|0.355|0.792|
| Run 3 | 7         | 0.01          | 0.700|0.533|0.626|

## Selected Best Model
Run 2

## Why
RMSE is the primary selection metric, and Run 2 has the lowest validation RMSE (0.522),
meaning the smallest typical prediction error. It is also best on MAE (0.355) and R² (0.792),
so the choice does not depend on which metric is used.

Run 1 (depth 3, lr 0.1) is close but slightly worse, probably because its shallower trees
have less capacity to capture the data. Run 3 (depth 7, lr 0.01) is clearly the worst: with
only 100 boosting rounds, a learning rate of 0.01 is too slow for the model to converge,
so it underfits (R² of 0.626). Run 2's mix of moderate depth and moderate learning rate
gave the best balance.