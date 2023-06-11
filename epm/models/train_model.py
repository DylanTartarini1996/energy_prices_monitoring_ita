import matplotlib.pyplot as plt
import mlflow
import mlflow.xgboost
import pandas as pd
import subprocess
from mlflow.tracking import MlflowClient
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error

from utils.xgbforecaster import XGBForecaster


def train_model(
    experiment_name: str, train_data: pd.DataFrame, test_data: pd.DataFrame
) -> XGBRegressor:
    # prepare train and test data

    test_data.fillna(train_data.iloc[-1, -1])
    X_train = train_data.iloc[:, :-1].values
    X_test = test_data.iloc[:, :-1].values
    y_train = train_data.iloc[:, -1].values
    y_test = test_data.iloc[:, -1].values

    # n-folds
    effective_df_length = len(train_data) - len(test_data)
    max_folds = effective_df_length // len(test_data)
    n_folds = min(max_folds, 10)

    # Create the experiment if it does not exist
    client = MlflowClient()
    experiment = mlflow.get_experiment_by_name(experiment_name)
    experiment_id = experiment.experiment_id
    latest_run = client.search_runs(
        experiment_id, order_by=["start_time desc"], max_results=1
    )[0]
    # if experiment is None:
    #     mlflow.create_experiment(experiment_name)
    #     experiment = mlflow.get_experiment_by_name(experiment_name)

    # enable auto logging
    mlflow.xgboost.autolog()

    with mlflow.start_run(run_id=latest_run.info.run_id):
        # log the script
        mlflow.log_artifact(__file__)

        # Get current commit hash
        commit_hash = (
            subprocess.check_output(["git", "rev-parse", "HEAD"])
            .strip()
            .decode("utf-8")
        )
        # Log Git commit hash as a parameter
        mlflow.log_param("commit_hash", commit_hash)

        xgb = XGBRegressor()
        parameters_xgb = {
            "gamma": [0, 30, 100, 200],
            "eta": [0.3, 0.03, 0.003],
            "max_depth": [6, 12, 30],
        }
        xgb_grid, predictions_xgb = XGBForecaster.grid_search(
            xgb,
            parameters_xgb,
            n_folds,
            train_data,
            len(test_data),
            n_jobs=-1,
            verbose=1,
        )
        mae = mean_absolute_error(y_test, predictions_xgb)
        mape = mean_absolute_percentage_error(y_test, predictions_xgb)

        # log metrics
        mlflow.log_metrics({"MAE": mae, "MAPE": mape})

        return xgb
