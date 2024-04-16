# COVID-19-s-Impact-on-Stock-Prices

# How to run

For linear regression model (predict % of change in stocks price from number of new COVID-19 case)  

`python3 covidPredictorV1.py`

For linear regression model and SVM (predict stocks price from number of new COVID-19 case)  

`python3 covidPredictor.py`

# Description
The program will read a default set of data(*.csv) from `./RawData` and use it for model training and prediction.

The program uses the following libraries:
* pandas
* numpy
* sklearn
* matplotlib

Data to be read by the program are stored under `./RawData`  
Data produced by the program are stored under `./data`


### Raw data are taken from the following sources:
* Stocks Data：https://ca.investing.com/equities/bce-historical-data
* COIVD-19 Data：https://ourworldindata.org/coronavirus-source-data


### All the graphs in the report are produced by the program.
