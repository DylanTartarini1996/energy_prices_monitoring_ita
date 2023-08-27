import pandas as pd
import numpy as np
import mlflow
import mlflow.xgboost
from xgboost import XGBModel, XGBRegressor
from sklearn.model_selection import GridSearchCV
import subprocess
from mlflow.tracking import MlflowClient
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error
from typing import Tuple

from epm.models.utils.preprocessing import Preprocessing


class XGBForecaster:
    """
    XGBoost model used for univariate or multivariate forecasting.
    """

    def __init__(self) -> None:
        self.xgb = XGBRegressor()

    def fit(self, model: XGBRegressor, train_ensamble: pd.DataFrame) -> XGBRegressor:
        """
        Trains an XGBRegressor on a TimeSeries Dataset
        
        Returns
        ---------
        model: XGBRegressor
            a fitted instance of the XGBRegressor
        """
        data = np.asarray(train_ensamble)
        X, y = data[:, :-1], data[:, -1]
        model.fit(X, y)
        return model

    def forecast(self, 
                 model: XGBRegressor, 
                 row_just_before: int, 
                 steps_ahead: int
        ) -> list:
        """
            Rolling prediction with the model_fitted for predicting n=steps_ahead new instances.
            This instances will immediately follow row_just_before, which is the last row of the dataframe available
        """
        row_just_before = np.asarray(row_just_before)[1:]
        current_row = row_just_before.reshape(1, -1)
        forecast = []
        for _ in range(steps_ahead):
            pred = model.predict(current_row)
            forecast.append(pred[0])
            current_row = np.concatenate((current_row[0][1:], pred)).reshape(1, -1)
        return forecast

    def grid_search(
        self, parameters, n_folds, train_df, test_size, n_jobs=1, verbose=0
    ):
        model = self.xgb
        grid = GridSearchCV(
            model, parameters, cv=n_folds, n_jobs=n_jobs, verbose=verbose
        )
        grid = XGBForecaster.fit(self, model=grid, train_ensamble=train_df)
        predictions = XGBForecaster.forecast(
            self, 
            model=grid,
            row_just_before=train_df.iloc[-1, :], 
            steps_ahead=test_size
        )
        return grid, predictions

    def preprocess(self, data: pd.DataFrame, col: str, experiment_name: str, frac: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
            Creates an experiment run for the model to be trained and preprocess the data

        Parameters
        ----------
        data: pd.DataFrame
            data to use for training.
        col: str
            column to be kept in the data
        experiment_name: str
            name of the experiment for training the model; might refer to the commodity to forecast.
        frac: float
            percentage of data to hold out for testing the model.

        """
        # Create the experiment if it does not exist
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            mlflow.create_experiment(experiment_name)
            experiment = mlflow.get_experiment_by_name(experiment_name)

        with mlflow.start_run(experiment_id=experiment.experiment_id):
            # logging information on input data
            data = Preprocessing.preprocessing(data, col)

            train, test = Preprocessing.train_test_split_df(
                data=data, n_test=round(len(data) * frac)
            )

            proc_training_data = Preprocessing.series_to_supervised(
                data=train, n_in=1, dropnan=True
            )
            proc_testing_data = Preprocessing.series_to_supervised(
                data=test, n_in=1, dropnan=False
            )

            mlflow.log_param(key="pct_data_for_training", value=(1 - frac))
            mlflow.log_param(key="pct_data_for_testing", value=(frac))

            return proc_training_data, proc_testing_data
        
    def train_model(
        self, experiment_name: str, train_data: pd.DataFrame, test_data: pd.DataFrame
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
            # mlflow.log_param("commit_hash", commit_hash)

            parameters_xgb = {
                "gamma": [0, 30, 100, 200],
                "eta": [0.3, 0.03, 0.003],
                "max_depth": [6, 12, 30],
            }
            xgb_grid, predictions_xgb = XGBForecaster.grid_search(
                self,
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

        return xgb_grid

