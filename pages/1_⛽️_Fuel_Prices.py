import datetime
import pandas as pd 
import plotly.express as px
import streamlit as st

from epm.scraping_utils.fuel_prices import FuelPrices

st.set_page_config(
    page_title="Prezzi Carburanti",
    page_icon="📈⛽️",
)

st.markdown(
    """
        I dati mostrati in questa pagina rappresentano l'
        **andamento dei prezzi dei carburanti** dal 2005 alla data corrente 
        e sono una rielaborazione di quelli forniti dal 
        [Ministero dell'Ambiente e della Sicurezza Energetica](https://dgsaie.mise.gov.it/open-data).
    """
)

fp = FuelPrices()

@st.cache_data
def get_fuel_prices() -> pd.DataFrame:
    fuel_prices=fp.get_data()
    return fuel_prices

fuel_prices = get_fuel_prices()

with st.expander(label='Fuel Prices Data'):
    st.dataframe(data=fuel_prices, use_container_width=True)

with st.container():
    fig = px.line(data_frame=fuel_prices,
                  y=[fuel_prices["BENZINA"], fuel_prices["DIESEL"], fuel_prices["GPL"]])
    st.plotly_chart(fig)