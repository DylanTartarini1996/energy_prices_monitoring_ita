import pandas as pd
import requests


class FuelPrices:
    """
    Weekly Fuel Prices in Italy since 2005. Prices listed include
    * gasoline
    * diesel
    * nlg
    """

    def __init__(self):
        # net of VAT average fuel prices in Italy
        self.url = "https://dgsaie.mise.gov.it/open_data_export.php?export-id=1&amp;export-type=csv"

    def get_data(self) -> pd.DataFrame:
        # Send a GET request to the URL
        response = requests.get(self.url)
        # Check if the request was successful
        if response.status_code == 200:
            # Save the response content to a file
            with open(
                "fuel_prices.csv", "wb"
            ) as f:
                f.write(response.content)
            print("CSV file downloaded successfully.")
        else:
            print("Failed to download CSV file.")
        fuel_prices = pd.read_csv(
            "fuel_prices.csv", index_col=0
        )
        fuel_prices = fuel_prices.iloc[:, 0:3]
        fuel_prices = fuel_prices.rename(columns={"GASOLIO_AUTO": "DIESEL"})
        # 1€ per liter
        fuel_prices = fuel_prices.div(1000)
        self.df = fuel_prices
        return self.df
