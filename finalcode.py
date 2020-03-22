import warnings
warnings.filterwarnings('ignore')
import os
import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
#%matplotlib inline

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LinearRegression

import tensorflow as tf
from tensorflow import keras


from sklearn import preprocessing


#os.chdir("C:\Users\HP\Desktop\Project\minor2") #for local computer
weather_df=pd.read_csv('weatherHistory.csv')


weather_df.head()


weather_df.columns  #Gives an array of coulamn names


weather_df.shape    #prints the shape(no of rows and colums)of dataframe


weather_df.describe()   #generates descriptive statistics that summarise the central
                        #tendency,dispersion and shape of data excluding null(NaN)values

weather_df.info()   #prints a concise summary of a dataframe


weather_df.isnull().any()   #checks if any null value in a paricular column


weather_df.isnull().all()   #if all values of a column are null


round(100*(weather_df.isnull().sum()/len(weather_df.index)),2)


weather_df['Precip Type'].value_counts()    #2 type of precipitation rain and snow
                                            #other are null

weather_df.loc[weather_df['Precip Type'].isnull(),'Precip Type']='rain'


round(100*(weather_df.isnull().sum()/len(weather_df.index)),2)    #Now is zero due to
                                                        #value get replaced by rain

weather_df.loc[weather_df['Precip Type']=='rain','Precip Type']=1
weather_df.loc[weather_df['Precip Type']=='snow','Precip Type']=0


weather_df_num=weather_df[list(weather_df.dtypes[weather_df.dtypes!='object'].index)]


weather_y=weather_df_num.pop('Temperature (C)')
weather_x=weather_df_num


train_x,test_x,train_y,test_y=train_test_split(weather_x,weather_y,test_size=0.2,random_state=4)


train_x.head()

#Starting of linear Regression model
model=LinearRegression()
model.fit(train_x,train_y)


prediction=model.predict(test_x)


np.mean((prediction-test_y)**2)


pd.DataFrame({'actual':test_y,
              'prediction':prediction,
              'diff':(test_y-prediction)})
#end of linear Regression model


#start of Polynomial Regression

from sklearn.preprocessing import PolynomialFeatures


poly=PolynomialFeatures(degree=4)
x_poly=poly.fit_transform(train_x)
poly.fit(x_poly,train_y)
lin2=LinearRegression()
lin2.fit(x_poly,train_y)


prediction2=lin2.predict(poly.fit_transform(test_x))
#calculating error
np.mean((prediction2-test_y)**2)


pd.DataFrame({'actual':test_y,
              'prediction':prediction2,
              'diff':(test_y-prediction2)})
#end of Polynomial Regression


#start of DecisionTreeRegressor
from sklearn.tree import DecisionTreeRegressor
regressor=DecisionTreeRegressor(random_state=0)
regressor.fit(train_x,train_y)


prediction3=regressor.predict(test_x)
np.mean((prediction3-test_y)**2)


pd.DataFrame({'actual':test_y,
              'prediction':prediction3,
              'diff':(test_y-prediction3)})
#end of DecisionTreeRegressor


#start of RandomForestRegressor
from sklearn.ensemble import RandomForestRegressor
regr=RandomForestRegressor(max_depth=10,random_state=0,n_estimators=100)
regr.fit(train_x,train_y)


prediction4=regr.predict(test_x)
np.mean((prediction4-test_y)**2)


pd.DataFrame({'actual':test_y,
              'prediction':prediction4,
              'diff':(test_y-prediction4)})
#end of RandomForestRegressor


#start of RandomForestRegressor 2
from sklearn.ensemble import RandomForestRegressor
regr=RandomForestRegressor(max_depth=50,random_state=0,n_estimators=100)
regr.fit(train_x,train_y)


prediction5=regr.predict(test_x)
np.mean((prediction5-test_y)**2)


pd.DataFrame({'actual':test_y,
              'prediction':prediction5,
              'diff':(test_y-prediction5)})
#end of RandomForestRegressor 2



