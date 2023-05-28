import pandas as pd
import plotly.express as px
import streamlit as st

from scraping_utils.gas_prices import GasPrices

st.set_page_config(
    page_title="Prezzo del Gas Naturale",
    page_icon="📈🔥",
)

st.markdown(
    """
    Al contrario del prezzo dell'energia elettrica, quello del gas non è un prezzo unico:  
    infatti, non tutte le compagnie presenti nel business della vendita di gas naturale
    ai consumatori in Italia si riforniscono sulla stessa "piazza".
    I dati qui presentati sono quelli del mercato olandese [TTF](https://www.enel.it/en/supporto/faq/ttf-gas),
    ottenuti tramite l'API di Yahoo Finance ([yfinance](https://pypi.org/project/yfinance/)).  
    Con il tempo, il TTF ha assunto il ruolo di indice di riferimento per il prezzo del gas naturale nel mercato europeo.
    """
)


@st.cache_data()
def get_gas_prices() -> pd.DataFrame:
    gp = GasPrices.get_data()
    return gp

gas_prices = get_gas_prices()

fig = px.line(gas_prices, y="GAS NATURALE")

st.plotly_chart(fig)
