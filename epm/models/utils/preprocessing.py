import numpy as np
import pandas as pd


class Preprocessing:
    def get_data(path: str) -> pd.DataFrame:
        data = pd.read_csv(path, index_col=0)
        return data

    def preprocessing(data: pd.DataFrame, col: str) -> pd.DataFrame:
        """
        Returns data with index and frequency of index set

        Parameters
        ----------
        data: pd.DataFrame

        col: str
            name of the column that will be kept
        """
        data.index = pd.to_datetime(data.index)
        data = data[col]
        data = data.div(1000)
        data.index.freq = pd.infer_freq(data.index)
        return data

    def train_test_split_series(data: pd.DataFrame, n_test: int) -> pd.DataFrame:
        return data.iloc[:-n_test], data.iloc[-n_test:]

    def train_test_split_df(data: pd.DataFrame, n_test: int) -> pd.DataFrame:
        return data.iloc[:-n_test], data.iloc[-n_test:]

    def series_to_supervised(
        data: pd.Series, n_in: int = 1, dropnan: bool = True
    ) -> np.array:
        """
        Converts a sequence of numbers, i.e. a univariate time series, into a matrix
        with one array (series at time t) plus one more array for each n_in
        (lags at times t-1, t-2, .., t-n_in).

        Parameters
        ----------
        data: pd.Series

        n_in: int
            number of lags to create from the original series.
            For each lag required, one more column will be added,
            at the cost of one row of observations.

        dropnan: bool

        """
        df = pd.DataFrame(data)
        cols = list()
        # input sequence (t-n, ... t-1)
        for i in range(n_in, 0, -1):
            cols.append(df.shift(i))
        cols.append(df)
        # put it all together
        agg = pd.concat(cols, axis=1)
        # drop rows with NaN values (in particular the first and the last rows)
        if dropnan:
            agg.dropna(inplace=True)

        return agg
