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

if "target_selected" not in st.session_state:
    st.session_state.target_selected = False

def set_experiment():
    """
    Depending on the chosen fuel, sets an experiment for mlflow, dropping the 
    columns that are not of interest for the model. 
    It also sets to True the status of the selectbox.
    """
    if st.session_state["target_col"] == "BENZINA":
        st.session_state = "gasoline_model" 
        artifact_path = "gasoline_prices_model"
    elif st.session_state["target_col"] == "DIESEL":
        experiment_name = "diesel_model" 
        artifact_path = "diesel_prices_model"
    elif st.session_state["target_col"] == "GPL":
        experiment_name = "nlg_model" 
        artifact_path = "nlg_prices_model"

    st.session_state["target_selected"] = True

with st.expander(label='Fuel Prices Data'):
    st.dataframe(data=fuel_prices, use_container_width=True)

with st.container():
    if not st.session_state["target_selected"]:
        fig = px.line(
            data_frame=fuel_prices,
            y=[fuel_prices["BENZINA"], fuel_prices["DIESEL"], fuel_prices["GPL"]],
            title="Andamento storico dei prezzi dei carburanti"
        )
        st.plotly_chart(fig)
        st.write("Puoi selezionare i dati di un carburante per addestrare un modello e effettuare previsioni")
    else:
        fig = px.line(
            data_frame=fuel_prices,
            y=st.session_state["target_col"],
            title=f'Andamento storico dei prezzi {st.session_state["target_col"]}'
        )
        st.plotly_chart(fig)
        st.write(st.session_state.experiment_name)

if not st.session_state["target_selected"]:
    st.session_state["target_col"] = st.selectbox(
        label="Seleziona il carburante di cui vuoi prevedere l'andamento del prezzo",
        options=("BENZINA", "DIESEL", "GPL"),
        index=None,
        placeholder="Seleziona dati..",
        on_change=set_experiment
    )

