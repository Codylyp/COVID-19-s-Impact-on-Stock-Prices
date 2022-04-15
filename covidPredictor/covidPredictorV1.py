# 4710 Group Project
# This program uses the following libraries:
#  * pandas
#  * numpy
#  * sklearn
#  * matplotlib

import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt

from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


# show the relationship between new COVID-19 cases and the Canadian Airline stock price changes
def predict_All_stock(data_path, x, y1, y2, label, **y3):
    
    # read data file
    data = pd.read_csv(data_path)
    
    # preprocess data
    data = data[~data[x].isin([0])]  # remove rows with newCases = 0
    #data = data[54:]
    data = data.fillna(data.mean())  # fill all NA value with the mean value of this colume
        
    # define daily new cases as the feature
    new_cases_X = data[x].values.reshape(-1,1)
    #print(data[x])

    # define the stock price of Canada Airline change as target
    if ('optional' in y3):
        data['average']=(data[y1]+data[y2]+data[y3['optional']])/3
    else:
        data['average']=(data[y1]+data[y2])/2
    #print(data["average"])
    stock_change_y = data['average']
    
    # split dataset into the training and test data
    X_train, X_test, y_train, y_test = train_test_split(new_cases_X, stock_change_y, test_size=0.2)

    # create a linear regression object
    lin_reg = linear_model.LinearRegression()
    
    # training the model
    lin_reg.fit(X_train, y_train)

    # test the model
    stock_price_predict = lin_reg.predict(X_test)

    #model performance
    print('MODEL PERFORMANCE FOR: %s' % label)
    print('INTERCEPTION:%.2f' % lin_reg.intercept_)
    print('COEFFICIENTS:', lin_reg.coef_)
    print('MEAN SQUARED ERROR: %.2f' % mean_squared_error(y_test.values, stock_price_predict))
    print('COEF. OF DETERMINATION: %.2f' % r2_score(y_test.values, stock_price_predict))

    #print accuracy
    print('score: %.3f\n' % lin_reg.score(X_test,y_test))

    #draw the scatter plot
    plt.xlabel(x)
    plt.ylabel(label + ' stock price changes')
    plt.title('relationship between new COVID-19 cases and ' + label + ' stock price changes')
    plt.scatter(X_test, y_test, color = 'red')
    plt.plot(X_test, stock_price_predict, color = 'blue', linewidth = 1.5)
    plt.show()


# show the relationship between new COVID-19 cases and one specific Canadian stock price changes
def predict_stock(data_path, x, y):
    
    # read data file
    data = pd.read_csv(data_path)
    
    # preprocess data
    data = data[~data[x].isin([0])]    # remove rows with newCases = 0
    #data = data[54:]
    data = data.fillna(data.mean())     # fill all NA value with the mean value of this colume
        
    # define daily new cases as the feature
    new_cases_X = data[x].values.reshape(-1,1)
    #print(data[x])

    # define stock price change as target
    stock_change_y = data[y]
    #print(data[y])

    # split dataset into the training and test data
    X_train, X_test, y_train, y_test = train_test_split(new_cases_X, stock_change_y, test_size=0.2)

    # create a linear regression object
    lin_reg = linear_model.LinearRegression()
    
    # training the model
    lin_reg.fit(X_train, y_train)

    # test the model
    stock_price_predict = lin_reg.predict(X_test)

    #model performance
    print('MODEL PERFORMANCE FOR: %s' % y)
    print('INTERCEPTION:%.2f' % lin_reg.intercept_)
    print('COEFFICIENTS:', lin_reg.coef_)
    print('MEAN SQUARED ERROR: %.2f' % mean_squared_error(y_test.values, stock_price_predict))
    print('COEF. OF DETERMINATION: %.2f' % r2_score(y_test.values, stock_price_predict))

    #print accuracy
    print('score: %.3f\n' % lin_reg.score(X_test,y_test))

    #draw the scatter plot
    plt.xlabel(x)
    plt.ylabel(y + ' stock price changes')
    plt.title('relationship between new COVID-19 cases and ' + y + ' stock price changes')
    plt.scatter(X_test, y_test, color = 'red')
    plt.plot(X_test, stock_price_predict, color = 'blue', linewidth = 1.5)
    plt.show()


# Program starts here.......
#  show the relationship between new COVID-19 cases and one specific Canadian stock price changes
predict_stock('./RawData/mergedTable/mergedAirlinesCA.csv', 'newCases', 'airCanada')
predict_stock('./RawData/mergedTable/mergedAirlinesCA.csv', 'newCases', 'cargoJest')
predict_stock('./RawData/mergedTable/mergedCommCA.csv', 'newCases', 'bell')
predict_stock('./RawData/mergedTable/mergedCommCA.csv', 'newCases', 'rogers')
predict_stock('./RawData/mergedTable/mergedCommCA.csv', 'newCases', 'shaw')
predict_stock('./RawData/mergedTable/mergedRetailCA.csv', 'newCases', 'canadianTire')
predict_stock('./RawData/mergedTable/mergedRetailCA.csv', 'newCases', 'dollarama')
predict_stock('./RawData/mergedTable/mergedRetailCA.csv', 'newCases', 'loblaws')

#  show the relationship between new COVID-19 cases and one specific USA stock price changes
predict_stock('./RawData/mergedTable/mergedAirlinesUS.csv', 'newCases', 'americanAirline')
predict_stock('./RawData/mergedTable/mergedAirlinesUS.csv', 'newCases', 'unitedAirline')
predict_stock('./RawData/mergedTable/mergedCommUS.csv', 'newCases', 'at&t')
predict_stock('./RawData/mergedTable/mergedCommUS.csv', 'newCases', 'verizon')
predict_stock('./RawData/mergedTable/mergedCommUS.csv', 'newCases', 'tMobile')
predict_stock('./RawData/mergedTable/mergedRetailUS.csv', 'newCases', 'amazon')
predict_stock('./RawData/mergedTable/mergedRetailUS.csv', 'newCases', 'target')
predict_stock('./RawData/mergedTable/mergedRetailUS.csv', 'newCases', 'walmart')

# show the relationship between new COVID-19 cases and the three kind of Canadian general stock price changes
predict_All_stock('./RawData/mergedTable/mergedAirlinesCA.csv', 'newCases', 'airCanada', 'cargoJest', 'Canadian Airline Industry')
predict_All_stock('./RawData/mergedTable/mergedCommCA.csv', 'newCases', 'bell', 'rogers', 'Canadian Telecom Industry ', optional='shaw')
predict_All_stock('./RawData/mergedTable/mergedRetailCA.csv', 'newCases', 'canadianTire', 'dollarama', 'Canadian Retail Industry', optional='loblaws')

# show the relationship between new COVID-19 cases and the three kind of Canadian general stock price changes
predict_All_stock('./RawData/mergedTable/mergedAirlinesUS.csv', 'newCases', 'unitedAirline', 'americanAirline', 'USA Airline Industry')
predict_All_stock('./RawData/mergedTable/mergedCommUS.csv', 'newCases', 'at&t', 'verizon', 'USA Telecom Industry', optional='tMobile')
predict_All_stock('./RawData/mergedTable/mergedRetailUS.csv', 'newCases', 'amazon', 'target', 'USA Retail Industry', optional='walmart')

print('\n Program exited normally \n')

