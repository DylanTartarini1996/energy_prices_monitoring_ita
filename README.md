# Energy Prices Monitoring ITA  

### To-do list (Last Update 04/04/2023)
* include Gas Prices
* forecasting algo for EE
* forecasting algo for Fuel(s)

## Project Description
Streamlit Application built for monitoring energy prices in Italy across time.  
Italy is not self-sufficient in terms of energy provisioning, but many of its companies are energy-intensive and need to look very carefully at their electricity bill, as well as their transportation/delivery costs; their heating/air conditioning needs and prices associated with them are also not to be underestimated in a period of economic crisis.  

## Data Collection
Data for this application are downloaded from a variety of sources; they include
* **Fuel prices** from [Ministero dell'Ambiente e della Sicurezza Energetica](https://dgsaie.mise.gov.it/open-data);
* **Electricity prices** from [GME](https://www.mercatoelettrico.org/it/)

## Analysis
Each page of the app collects data for a specific subset of energy prices, from a different data source. On each dataset are performed
* simple plotting;
* **Exploratory Data Analysis**, including *Seasonal Decomposition*;
* **Forecasting** (autoregressive);
Please note that forecast results are NOT a guarantee that market prices will eventually be exactly turn out that way, and are only presented as orientative values.