import pandas as pd
import plotly.express as px
import streamlit as st

from epm.models.xgbforecaster import XGBForecaster

from epm.scraping_utils.gas_prices import GasPrices

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

@st.cache_resource()
def forecaster_init() -> XGBForecaster:
    forecaster = XGBForecaster()
    return forecaster
forecaster = forecaster_init()

with st.sidebar:
    frac = st.slider(
        label="Percentuale dello storico da utilizzare per l'addestramento del modello",
        min_value=0.01,
        max_value=1.0,
        value=0.2
        )
    train = st.button(label="Addestra il modello!", type="primary")

with st.expander(label="Gas Naturale (TTF)"):
    st.dataframe(data=gas_prices, use_container_width=True)

with st.container():
    fig = px.line(gas_prices, y="GAS NATURALE")
    st.plotly_chart(fig)

if train:

    experiment_name = "gas_model"

    with st.spinner("Addestramento del modello in corso.."):
        train_data, test_data = forecaster.preprocess(
            data=gas_prices,
            col="GAS NATURALE",
            experiment_name=experiment_name,
            frac=frac
        )

        model = forecaster.train_model(
            experiment_name=experiment_name,
            train_data=train_data,
            test_data=test_data
        )
    st.success('Fatto! Il modello è addestrato e pronto ad effettuare le sue predizioni!')
else: 
    st.write("Puoi addestrare un algoritmo su questi dati cliccando sul bottone a sinistra!")
