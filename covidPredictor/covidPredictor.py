# 4710 Group Project
# Analysis and predict the relationship between stock price and COVID-19 new cases
# This program uses the following libraries:
#  * pandas
#  * sklearn
#  * matplotlib

# imports
import os
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import linear_model, preprocessing
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


# calculate index for a list of stock price
def gen_stocks_index(stocks, covid, merged_path):
    
    covid_dp = lambda x: pd.datetime.strptime(x, "%Y-%m-%d")
    stock_dp = lambda x: pd.datetime.strptime(x, "%b %d, %Y")
    covid_used_col = ['Date', 'newCases']
    stock_used_col = ['Date', 'Price']
    covid_df = pd.read_csv(covid, parse_dates=['Date'], date_parser=covid_dp, usecols=covid_used_col)
    stocks_dfs = []
    df = None

    # assign each stock to a data frame
    for stock in stocks:
        stocks_dfs.append(pd.read_csv(stock, thousands=",", parse_dates=['Date'] ,date_parser=stock_dp, usecols=stock_used_col))

    df = covid_df
    
    # left join all the stocks to the covid 
    for stock_df in stocks_dfs:
        df = pd.merge(df, stock_df, how='left', on="Date")

    # drop the rows that has no stock info (mostly b/c weekend)
    df.dropna(axis=0, subset=df.columns.difference(['Date', 'newCases']) ,how='all', inplace=True)

     # fill the non-trade day price with unchange price(price from yesterday)
    df = df.fillna(method='ffill')  
    
    # special case for shaw, the first day's price is missing, fill it with the next day's price
    df = df.fillna(method='backfill') 

    # calculate equal weighting index based on price of stocks
    indexs = [] # index column that will be append to the table
    index = 0 # initial index
    
    # initialize the index with the initial price for each stock
    for col in range(2, df.columns.size):
        init_price = df.iloc[:,col][df.iloc[:,col].first_valid_index()]
        index += init_price
        
    indexs.append(index)

    # traverse through each row and average the changes, update the index for that role based on the change
    row_number = 0
    prev_price_a = 0
    prev_price_b = 0
    prev_price_c = 0
    prev_index = indexs[0]
    
    if len(df.columns==5):
        # The index is consist of 2 stocks
        for index, row in df.iterrows():
            if(row_number == 0):
                prev_price_a = row[2]
                prev_price_b = row[3]
            else:
                index_change_rate = ((row[2]-prev_price_a)/prev_price_a + (row[3]-prev_price_b)/prev_price_b)/2
                new_index = prev_index + prev_index * index_change_rate
                indexs.append(new_index)
                prev_price_a = row[2]
                prev_price_b = row[3]
                prev_index = new_index
            row_number += 1      
    else:
        # The index is consist of 3 stocks
        for index, row in df.iterrows():
            if(row_number == 0):
                prev_price_a = row[2]
                prev_price_b = row[3]
                prev_price_c = row[4]
            else:
                index_change_rate = ((row[2]-prev_price_a)/prev_price_a + (row[3]-prev_price_b)/prev_price_b + (row[4]-prev_price_c)/prev_price_c)/3
                new_index = prev_index + prev_index * index_change_rate
                indexs.append(new_index)
                prev_price_a = row[2]
                prev_price_b = row[3]
                prev_price_c = row[4]
                prev_index = new_index
            row_number += 1        
    
    #append index column to the table and save the table
    df['index'] = indexs   
    df.to_csv(merged_path)
    

# generate data file
def gen_price_to_case(stock, covid, merged_p):
    covid_dp = lambda x: pd.datetime.strptime(x, "%Y-%m-%d")
    stock_dp = lambda x: pd.datetime.strptime(x, "%b %d, %Y")
    covid_df = pd.read_csv(covid, parse_dates=["Date"], date_parser=covid_dp)
    stock_df = pd.read_csv(stock, parse_dates=["Date"], date_parser=stock_dp)
    merged = pd.merge(covid_df, stock_df, on="Date")
    merged.to_csv(merged_p)


# use SVM to explore and predict the relationship between COVID-19 new cases and stock price
def predict_stock_svm(data_path, feature, target, stock_name):
    
    # build SVM model
    svr_poly = SVR(kernel='poly', C=1, gamma='auto', degree=3, epsilon=.1,
               coef0=1)
    
    # read data file
    df = pd.read_csv(data_path,thousands=',')

    # preprocess data
    df = df[df[feature]>0]    # remove rows with newCases = 0
    df = df.fillna(df.mean())     # fill all NA value with the mean value of this colume
    
    # choose features and targets
    feature_X = preprocessing.scale(df[feature].values.reshape(-1,1))
    target_y = preprocessing.scale(df[target])

    # split data into training set and test set in 80:20 ratio
    X_train, X_test, y_train, y_test = train_test_split(feature_X, target_y, test_size=0.2)

    # do the SVM predict
    predict = svr_poly.fit(feature_X, target_y).predict(feature_X)
    stock_price_predict = svr_poly.fit(feature_X, target_y).predict(X_test)

    #draw the scatter plot
    plt.xlabel('New Cases (scaled)')
    plt.ylabel('Stock Price(scaled)')
    plt.title('Relationship Between New COVID-19 Cases and ' + stock_name + ' Stock Price \n SVM - Data Scaled')
    plt.plot(feature_X, predict, color='blue', linewidth = 1.5)
    plt.scatter(X_test, y_test,marker = 'o', color = 'r')
    plt.show()
    
    #model performance
    print('MODEL PERFORMANCE FOR: %s' % stock_name)
    print('MEAN SQUARED ERROR: %.2f' % mean_squared_error(y_test, stock_price_predict))
    print('COEF. OF DETERATION: %.2f' % r2_score(y_test, stock_price_predict))

    #print accuracy
    print('score: %.3f\n' % svr_poly.score(X_test,y_test))


# use linear regression to explore and predict the relationship between new COVID-19 cases and stock price
def predict_stock(data_path, x, y, stock_name):
    
    # convert the data into a DataFrame object
    data = pd.read_csv(data_path,thousands=',')


    # preprocess data
    data = data[data[x]>0]    # remove rows with newCases = 0
    data = data.fillna(data.mean())     # fill all NA value with the mean value of this colume
        
    # define daily new cases as the feature
    new_cases_x = data[x].values.reshape(-1,1)
 
    # define stock price change as target
    stock_change_y = data[y]

    # split dataset into the training and test data
    x_train, x_test, y_train, y_test = train_test_split(new_cases_x, stock_change_y, test_size=0.2)

    # create a linear regression object
    lin_reg = linear_model.LinearRegression()
    
    # training the model
    lin_reg.fit(x_train, y_train)

    # test the model
    stock_price_predict = lin_reg.predict(x_test)

    #draw the scatter plot
    plt.xlabel('New Cases')
    plt.ylabel(stock_name + ' Stock Price')
    plt.title('Relationship Between New COVID-19 Cases and ' + stock_name + ' Stock Price')
    plt.scatter(x_test, y_test, color = 'red')
    plt.plot(x_test, stock_price_predict, color = 'blue', linewidth = 1.5)
    plt.show()
    
    #model performance
    print('MODEL PERFORMANCE FOR: %s' % stock_name)
    print('INTERCEPTION:%.2f' % lin_reg.intercept_)
    print('COEFFICIENTS:', lin_reg.coef_)
    print('MEAN SQUARED ERROR: %.2f' % mean_squared_error(y_test.values, stock_price_predict))
    print('COEF. OF DETERMINATION: %.2f' % r2_score(y_test.values, stock_price_predict))

    #print accuracy
    print('score: %.3f\n' % lin_reg.score(x_test,y_test))


# Definng path for data in different processing stages
raw_p = './RawData'
data_p = './data'
covid_p = raw_p + '/covids'
stock_p = data_p + '/stocks'
industry_p = data_p + '/industryIndex'

#create directories if it is not existed
if not os.path.exists(data_p):
    os.mkdir(data_p)
if not os.path.exists(stock_p):
    os.mkdir(stock_p)
if not os.path.exists(industry_p):
    os.mkdir(industry_p)

# define path of Covid-19 data of each contry
covid_us = covid_p + '/' + 'USnewCases.csv'
covid_ca = covid_p + '/' + 'CAnewCases.csv'

# define path of each raw stock data
amazon_r_p = raw_p + '/' + 'Amazon Historical Data.csv'
aairline_r_p = raw_p + '/' + 'American Airlines Historical Data.csv'
att_r_p = raw_p + '/' + 'AT&T.csv'
ac_r_p = raw_p + '/' + 'AirCanada.csv'
bell_r_p = raw_p + '/' + 'Bell Historical Data.csv'
ct_r_p = raw_p + '/' + 'Canadian Tire Historical Data.csv'
cj_r_p = raw_p + '/' + 'Cargojet.csv'
dollarm_r_p = raw_p + '/' + 'Dollarama Historical Data.csv'
loblaws_r_p = raw_p + '/' + 'Loblaws Historical Data.csv'
rogers_r_p = raw_p + '/' + 'Rogers.csv'
shaw_r_p = raw_p + '/' + 'Shaw.csv'
target_r_p = raw_p + '/' + 'Target Historical Data.csv'
uairline_r_p = raw_p + '/' + 'United Airlines Historical Data.csv'
walmart_r_p = raw_p + '/' + 'Walmart Historical Data.csv'
tmob_r_p = raw_p + '/' + 'T Mobile.csv'
verizon_r_p = raw_p + '/' + 'Verizon.csv'



# define path of newCase-Price of stock data
amazon_p = stock_p + 'amazon.csv'
aairline_p = stock_p + 'aairline.csv'
att_p = stock_p + 'att.csv'
ac_p = stock_p + 'ac.csv'
bell_p = stock_p + 'bell.csv'
ct_p = stock_p + 'ct.csv'
cj_p = stock_p + 'cj.csv'
dollarm_p = stock_p + 'dollarm.csv'
loblaws_p = stock_p + 'loblaws.csv'
rogers_p = stock_p + 'rogers.csv'
shaw_p = stock_p + 'shaw.csv'
target_p = stock_p + 'target.csv'
uairline_p = stock_p + 'uairline.csv'
walmart_p = stock_p + 'walmart.csv'
tmob_p = stock_p + 'tmob.csv'
verizon_p = stock_p + 'verizon.csv'



############################### Generate newCase-Price table for each stock from RawData #################################################

gen_price_to_case(amazon_r_p, covid_us, amazon_p)
gen_price_to_case(target_r_p, covid_us, target_p)
gen_price_to_case(walmart_r_p, covid_us, walmart_p)
gen_price_to_case(aairline_r_p, covid_us, aairline_p)
gen_price_to_case(uairline_r_p, covid_us, uairline_p)
gen_price_to_case(att_r_p, covid_us, att_p)
gen_price_to_case(tmob_r_p, covid_us, tmob_p)
gen_price_to_case(verizon_r_p, covid_us, verizon_p)

gen_price_to_case(dollarm_r_p, covid_ca, dollarm_p)
gen_price_to_case(loblaws_r_p, covid_ca, loblaws_p)
gen_price_to_case(ct_r_p, covid_ca, ct_p)
gen_price_to_case(ac_r_p, covid_ca, ac_p)
gen_price_to_case(cj_r_p, covid_ca, cj_p)
gen_price_to_case(bell_r_p, covid_ca, bell_p)
gen_price_to_case(rogers_r_p, covid_ca, rogers_p)
gen_price_to_case(shaw_r_p, covid_ca, shaw_p)


################################################################################################################################################

############################### Generate index for stocks in each industry (equal weighting index) #########################################

# List of stocks for each industry

airline_stocks_us = [aairline_r_p, uairline_r_p]
telecom_stocks_us = [att_r_p, tmob_r_p, verizon_r_p]
retail_stocks_us = [amazon_r_p, target_r_p, walmart_r_p]
airline_stocks_ca = [ac_r_p, cj_r_p]
telecom_stocks_ca = [bell_r_p, rogers_r_p, shaw_r_p]
retail_stocks_ca = [ct_r_p, dollarm_r_p, loblaws_r_p]


# output tables that have index of each industry to files under directory ./data/industryIndex

gen_stocks_index(retail_stocks_us, covid_us, industry_p+'/retailStocksIndexUS.csv')
gen_stocks_index(airline_stocks_us, covid_us, industry_p+'/airlineStocksIndexUS.csv')
gen_stocks_index(telecom_stocks_us, covid_us, industry_p+'/telecomStocksIndexUS.csv')
gen_stocks_index(retail_stocks_ca, covid_ca, industry_p+'/retailStocksIndexCA.csv')
gen_stocks_index(airline_stocks_ca, covid_ca, industry_p+'/airlineStocksIndexCA.csv')
gen_stocks_index(telecom_stocks_ca, covid_ca, industry_p+'/telecomStocksIndexCA.csv')

####################################################################################################################################################

####################################### Train model using LINEAR REGRESSION ##############################################################

# Train the model using price as the feature

predict_stock(ac_p,'newCases', 'Price', 'Air Canada')
predict_stock(cj_p,'newCases', 'Price', 'CargoJet')
predict_stock(bell_p,'newCases', 'Price', 'Bell')
predict_stock(rogers_p,'newCases', 'Price', 'Rogers')
predict_stock(shaw_p,'newCases', 'Price', 'Shaw')
predict_stock(ct_p,'newCases', 'Price', 'Canadian Tire')
predict_stock(dollarm_p,'newCases', 'Price', 'Dollarama')
predict_stock(loblaws_p,'newCases', 'Price', 'Loblaws')

predict_stock(industry_p+'/airlineStocksIndexCA.csv','newCases', 'index', 'Candianian Airline Industry Index')
predict_stock(industry_p+'/telecomStocksIndexCA.csv','newCases', 'index', 'Candianian Telecom Industry Index')
predict_stock(industry_p+'/retailStocksIndexCA.csv','newCases', 'index', 'Candianian Retail Industry Index')

predict_stock(aairline_p,'newCases', 'Price', 'American Airline')
predict_stock(uairline_p,'newCases', 'Price', 'United Airline')
predict_stock(att_p,'newCases', 'Price', 'AT&T')
predict_stock(verizon_p,'newCases', 'Price', 'Verizon')
predict_stock(tmob_p,'newCases', 'Price', 'T Mobile')
predict_stock(amazon_p,'newCases', 'Price', 'Amazon')
predict_stock(target_p,'newCases', 'Price', 'Target')
predict_stock(walmart_p,'newCases', 'Price', 'Walmart')

predict_stock(industry_p+'/airlineStocksIndexUS.csv','newCases', 'index', 'US Airline Industry Index')
predict_stock(industry_p+'/telecomStocksIndexUS.csv','newCases', 'index', 'US Telecom Industry Index')
predict_stock(industry_p+'/retailStocksIndexUS.csv','newCases', 'index', 'US Retail Industry Index')


#####################################################################################################################################################

####################################### Trainmodel using SUPPORT VECTOR REGRESSION ##############################################################

predict_stock_svm(ac_p,'newCases', 'Price', 'Air Canada')
predict_stock_svm(cj_p,'newCases', 'Price', 'CargoJet')
predict_stock_svm(bell_p,'newCases', 'Price', 'Bell')
predict_stock_svm(rogers_p,'newCases', 'Price', 'Rogers')
predict_stock_svm(shaw_p,'newCases', 'Price', 'Shaw')
predict_stock_svm(ct_p,'newCases', 'Price', 'Canadian Tire')
predict_stock_svm(dollarm_p,'newCases', 'Price', 'Dollarama')
predict_stock_svm(loblaws_p,'newCases', 'Price', 'Loblaws')

predict_stock_svm(industry_p+'/airlineStocksIndexCA.csv','newCases', 'index', 'Candianian Airline Industry Index')
predict_stock_svm(industry_p+'/telecomStocksIndexCA.csv','newCases', 'index', 'Candianian Telecom Industry Index')
predict_stock_svm(industry_p+'/retailStocksIndexCA.csv','newCases', 'index', 'Candianian Retail Industry Index')

predict_stock_svm(aairline_p,'newCases', 'Price', 'American Airline')
predict_stock_svm(uairline_p,'newCases', 'Price', 'United Airline')
predict_stock_svm(att_p,'newCases', 'Price', 'AT&T')
predict_stock_svm(verizon_p,'newCases', 'Price', 'Verizon')
predict_stock_svm(tmob_p,'newCases', 'Price', 'T Mobile')
predict_stock_svm(amazon_p,'newCases', 'Price', 'Amazon')
predict_stock_svm(target_p,'newCases', 'Price', 'Target')
predict_stock_svm(walmart_p,'newCases', 'Price', 'Walmart')

predict_stock_svm(industry_p+'/airlineStocksIndexUS.csv','newCases', 'index', 'US Airline Industry Index')
predict_stock_svm(industry_p+'/telecomStocksIndexUS.csv','newCases', 'index', 'US Telecom Industry Index')
predict_stock_svm(industry_p+'/retailStocksIndexUS.csv','newCases', 'index', 'US Retail Industry Index')

print('\n Program exited normally \n')
