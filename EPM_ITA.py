import streamlit as st

st.set_page_config(
    page_title="Hello",
    page_icon="👋",
)

st.write("# Benvenuti nell'app di monitoraggio dei prezzi dell'energia in Italia 👋🇮🇹")

st.sidebar.success("Selezionare una pagina.")

st.markdown(
    """
    Scorri le pagine per monitorare l'andamento dei prezzi di energia elettrica, gas naturale e carburante.
    """
)


