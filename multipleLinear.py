#backwards regression

import numpy as np #contains math tools that is used to contain any math models in the code
import matplotlib.pyplot as plt #used for pretty and trendy graphs
import pandas as pd #the OG LIBRARY used to import data sets


data = pd.read_csv('50_Startups.csv')

X = data.iloc[:, :-1].values #all the lines, all the columns except the last one
y = data.iloc[:, -1].values

#categorical data
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
#column transformer = transformer[( , , index you want encoded)], remainder always equals passthrough
ct = ColumnTransformer(transformers = [('encoder', OneHotEncoder(), [3])], remainder = 'passthrough')
X = np.array(ct.fit_transform(X))

#training and testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#training the multiple linear regression model on the training set
from sklearn.linear_model import LinearRegression
regression = LinearRegression()
regression.fit(X_train, y_train)

#prediction test results
y_pred = regression.predict(X_test)
np.set_printoptions(precision = 2) #numerical value with only two places after the decimal
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))
#comparing predicted and actual
#shows multiple linear regression is good for this dataset
