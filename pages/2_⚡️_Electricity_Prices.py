import altair as alt
import pandas as pd
import streamlit as st

from scraping_utils.elec_prices import ElectricityPrices

st.set_page_config(
    page_title="Prezzo Unico Nazionale",
    page_icon="📈⚡️",
)

st.markdown(
    """
        Il PUN (acronimo di Prezzo Unico Nazionale) è il prezzo di
        riferimento all'ingrosso dell’energia elettrica che viene acquistata 
        sul mercato della Borsa Elettrica Italiana (**IPEX - Italian Power Exchange**).  

        Il PUN rappresenta, la media pesata nazionale dei prezzi zonali di vendita 
        dell’energia elettrica per ogni ora e per ogni giorno. 
        Il dato nazionale è un importo che viene calcolato sulla media di diversi 
        fattori, e che tiene conto delle quantità e dei prezzi formati nelle diverse
        zone d’Italia e nelle diverse ore della giornata.   

        Il dato proposto nell'app è un'aggregazione dei prezzi orari sulle settimane. 
        La fonte del dato è il [Gestore dei Mercati Elettrici](https://www.mercatoelettrico.org/it/)
    """
)
ep = ElectricityPrices()

@st.cache_data
def get_electricity_prices() -> pd.DataFrame:
    pun_prices = ep.get_data()
    return pun_prices

pun_prices = get_electricity_prices()

with st.expander(label='Electricity Prices Data'):
    st.dataframe(data=pun_prices, use_container_width=True)

with st.container():
    chart = alt.Chart(ep.melt_for_altair()).mark_line().encode(
        x='DateTime:T',
        y='value:Q',
        color='series:N'
        )
    chart = chart.configure_axis(
        labelFontSize=14,
        titleFontSize=16
        )
    chart = chart.encode(
        x=alt.X('DateTime:T', title='Date'),
        y=alt.Y('value:Q', title='PUN (€/MWh)')
        )

    st.altair_chart(chart, use_container_width=True)