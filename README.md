# COVID-19-s-Impact-on-Stock-Prices

# Description
This research is a discussion of the role played by data mining to showcase the stock trends from 3 major industries, such as retail, aviation, and telecommunications, during the pandemic in North America.

# How to run

For linear regression model (predict % of change in stocks price from number of new COVID-19 case)  

`python3 covidPredictorV1.py`

For linear regression model and SVM (predict stocks price from number of new COVID-19 case)  

`python3 covidPredictor.py`

# Technology
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

# Content
[Research Report](https://github.com/Codylyp/COVID-19-s-Impact-on-Stock-Prices/blob/main/project-report.pdf)

# Conclusion
The evaluation results show that we use the Linear Regression algorithm to find the relationship between stock price changes and the number of new additions, and we have not seen much. However, when using the SVM algorithm, we found that both the United States and Canada have affected the aviation industry. However, large airlines such as Canada's Air Canada and the United States' United Airline have not significantly impacted. The price of stocks in the telecommunications and offline retail industries has not been affected much. Still, companies like T Mobile that have user bases worldwide have been affected to a certain extent.
