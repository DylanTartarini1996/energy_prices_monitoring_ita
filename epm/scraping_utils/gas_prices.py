import pandas as pd
import yfinance as yf


class GasPrices:
    """
    Makes use of the yfinance python module
    to scrape natural gas prices from the TTF market.
    """

    # def __init__(self) -> None:
    #     pass

    def get_data() -> pd.DataFrame:
        symbol = "TTF=F"
        ticker = yf.Ticker(symbol)
        gas_prices = ticker.history(
            interval="1wk",
            start="2005-01-01",
            end=None,
            actions=True,
            auto_adjust=True,
            back_adjust=False,
        )

        gas_prices.index = pd.to_datetime(gas_prices.index)
        gas_prices.index = gas_prices.index.date
        gas_prices = gas_prices[["Close"]]
        gas_prices = gas_prices.rename(columns={"Close": "GAS NATURALE"})

        return gas_prices
