# Energy Prices Monitoring ITA  

### To-do list (Last Update 04/04/2023)
- [x] include Gas Prices
- [x] use plotly for graphs 
- [ ] forecasting algo for EE
- [ ] forecasting algo for Fuel(s)
- [ ] forecasting algo for Gas
- [ ] requirements file

## Project Description
Streamlit Application built for monitoring energy prices in Italy across time.  
Italy is not self-sufficient in terms of energy provisioning, but many of its companies are energy-intensive and need to look very carefully at their electricity bill, as well as their transportation/delivery costs; their heating/air conditioning needs and prices associated with them are also not to be underestimated in a period of economic crisis.  
  
This Demo aims to showcase prices for a list of commodities, as well as a Machine Learning model to forecast their evolution; please note that forecast results are NOT a guarantee that market prices will eventually be exactly turn out that way, and are only presented as orientative values.

## Data Collection
Data for this application are downloaded from a variety of sources; they include
* **Fuel prices** from [Ministero dell'Ambiente e della Sicurezza Energetica](https://dgsaie.mise.gov.it/open-data);
* **Electricity prices** from [GME](https://www.mercatoelettrico.org/it/)
* **Gas Prices** from the [TTF index](https://www.enel.it/en/supporto/faq/ttf-gas), obtained through the [Yahoo Finance API](https://pypi.org/project/yfinance/).
