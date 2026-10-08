import mlflow
import mlflow.xgboost
import numpy as np
import xgboost as xgb
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

TRACKING_URI = "http://127.0.0.1:5000"
EXPERIMENT_NAME = "California_Housing_Optimization"
RANDOM_SEED = 42


def prepare_data():
    print("Fetching and splitting California Housing data...")
    california = fetch_california_housing()
    X, y = california.data, california.target
    # 80% training / 20% validation, reproducible via fixed seed
    return train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED)


def execute_run(run_name, max_depth, learning_rate, X_train, X_val, y_train, y_val):
    with mlflow.start_run(run_name=run_name):
        params = {
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "objective": "reg:squarederror",
            "eval_metric": "rmse",
            "seed": RANDOM_SEED,
        }
        mlflow.log_params(params)

        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)

        model = xgb.train(
            params=params,
            dtrain=dtrain,
            num_boost_round=100,
            evals=[(dtrain, "train"), (dval, "val")],
            verbose_eval=False,
        )

        preds = model.predict(dval)
        rmse = float(np.sqrt(mean_squared_error(y_val, preds)))
        mae = float(mean_absolute_error(y_val, preds))
        r2 = float(r2_score(y_val, preds))

        mlflow.log_metric("RMSE", rmse)
        mlflow.log_metric("MAE", mae)
        mlflow.log_metric("R2", r2)
        mlflow.xgboost.log_model(model, artifact_path="xgboost-model")

        print(f"{run_name} | depth={max_depth} lr={learning_rate} | "
              f"RMSE={rmse:.4f} MAE={mae:.4f} R2={r2:.4f}")


def main():
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    X_train, X_val, y_train, y_val = prepare_data()

    run_configs = [
        {"name": "Run 1", "depth": 3, "lr": 0.1},
        {"name": "Run 2", "depth": 5, "lr": 0.05},
        {"name": "Run 3", "depth": 7, "lr": 0.01},
    ]

    for cfg in run_configs:
        execute_run(cfg["name"], cfg["depth"], cfg["lr"],
                    X_train, X_val, y_train, y_val)


if __name__ == "__main__":
    main()
