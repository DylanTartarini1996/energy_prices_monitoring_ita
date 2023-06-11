import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV


class XGBForecaster:
    def fit(train_ensamble: pd.DataFrame, model: XGBRegressor) -> XGBRegressor:
        data = np.asarray(train_ensamble)
        X, y = data[:, :-1], data[:, -1]
        model.fit(X, y)
        return model

    def forecast(
        row_just_before: int, model_fitted: XGBRegressor, steps_ahead: int
    ) -> list:
        """
        Rolling prediction with the model_fitted for predicting n=steps_ahead new instances.
        This instances will immediately follow row_just_before, which is the last row of the dataframe available
        """
        row_just_before = np.asarray(row_just_before)[1:]
        current_row = row_just_before.reshape(1, -1)
        forecast = []
        for _ in range(steps_ahead):
            pred = model_fitted.predict(current_row)
            forecast.append(pred[0])
            current_row = np.concatenate((current_row[0][1:], pred)).reshape(1, -1)
        return forecast

    def grid_search(
        model, parameters, n_folds, train_df, test_size, n_jobs=1, verbose=0
    ):
        grid = GridSearchCV(
            model, parameters, cv=n_folds, n_jobs=n_jobs, verbose=verbose
        )
        grid = XGBForecaster.fit(train_df, grid)
        predictions = XGBForecaster.forecast(train_df.iloc[-1, :], grid, test_size)
        return grid, predictions
