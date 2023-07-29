import os
import sys
import mlflow
import pandas as pd

from typing import Tuple

from epm.models.utils.preprocessing import Preprocessing

def preprocess(
        data: pd.DataFrame, 
        col: str, 
        experiment_name: str,
        frac: float
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
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


# if __name__ == "__main__":
#     experiment_name = "xgboost_predictor"
#     path = "../poc_forecasting/data/fuel_prices.csv"
#     train_path="../poc_forecasting/data/train.csv"
#     test_path="../poc_forecasting/data/test.csv"
#     fuel="BENZINA"
#     frac = 0.2
#     preprocess(experiment_name=experiment_name,
#                filepath=path,
#                train_path=train_path,
#                test_path=test_path,
#                frac=frac)
