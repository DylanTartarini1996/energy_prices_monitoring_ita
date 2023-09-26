import pandas as pd
import requests
from io import BytesIO
from zipfile import ZipFile
from urllib.request import urlopen


class ElectricityPrices:
    def __init__(self):
        # net of VAT average fuel prices in Italy
        self.zip_url = (
            "https://www.mercatoelettrico.org/it/MenuBiblioteca/Documenti/Anno2023.zip"
        )

    def get_data(self) -> pd.DataFrame:
        """
        Fetches historical weekly data and joins them with
        the current year's weekly prices
        """
        hist_df = self.get_hist_data()
        new_data = self.get_new_data()
        # checking last date in history is not the same as the first in the new data
        if hist_df.iloc[-1].name == new_data.iloc[0].name:
            hist_df = hist_df[:-1]
            pun_prices = pd.concat([hist_df, new_data], axis=0)
        else:
            pun_prices = pd.concat([hist_df, new_data], axis=0)

        pun_prices.index = pd.to_datetime(pun_prices.index)
        pun_prices.index = pun_prices.index.date

        self.df = pun_prices
        
        return self.df

    def get_hist_data(self):
        hist_df = pd.read_csv(
            "data/hist_pun.csv",
            index_col=0,
            parse_dates=True,
        )

        return hist_df

    def get_new_data(self):
        """
        GME publish one xslx file per year, containing hourly prices.
        However, what we want is to have weekly prices for the current year
        """
        # year 2023
        resp = urlopen(self.zip_url)
        myzip = ZipFile(BytesIO(resp.read()))
        df = pd.read_excel(myzip.open(myzip.namelist()[0]), sheet_name=1)
        df = df.iloc[:, :3]
        df.columns = ["Date", "Hour", "PUN"]
        df["Date"] = df["Date"].astype(str)
        df["Date"] = pd.to_datetime(df["Date"])
        df["Hour"] = pd.to_datetime(df["Hour"], unit="h").dt.strftime("%H:%M")
        df["DateTime"] = pd.to_datetime(
            df["Date"].astype(str) + " " + df["Hour"].astype(str)
        )
        df.set_index(df["DateTime"], inplace=True, drop=True)
        df = df["PUN"]
        df = pd.DataFrame(df.resample("W").mean())
        return df
