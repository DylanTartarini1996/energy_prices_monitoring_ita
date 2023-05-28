import altair as alt
import pandas as pd 
import streamlit as st

from scraping_utils.fuel_prices import FuelPrices

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
    chart = alt.Chart(fp.melt_for_altair()).mark_line().encode(
        x='DATA:T',
        y='value:Q',
        color='series:N'
        )
    chart = chart.configure_axis(
        labelFontSize=14,
        titleFontSize=16
        )
    chart = chart.encode(
        x=alt.X('DATA:T', title='Date'),
        y=alt.Y('value:Q', title='Prices (€/lt)')
        )

    st.altair_chart(chart, use_container_width=True)