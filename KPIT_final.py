# -*- coding: utf-8 -*-
"""
Created on Mon Jan  1 20:33:15 2017

This python file predicts passenger demand data from a given dataset

@author: Sahil Basera
"""
from  sklearn import preprocessing
from sklearn.cross_validation import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_regression
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor 



#To encode the data
le = preprocessing.LabelEncoder()

#to read csv file
data = pd.read_csv("KPIT_bus.csv")

#features
X = data[['date' , 'time' , 'stop' , 'day']]

#label
Y = data['Population']

#le= LabelEncoder()
le= preprocessing.LabelEncoder()
le.fit(["8:00" , "12:30" , "17:00" ])
for col in X :
   if col == "time" :
       X[col] = le.transform(X[col])

le1 = preprocessing.LabelEncoder()
le1.fit(["Sunday" , "Monday" , "Tuesday" , "Wednesday" , "Thursday" , "Friday" , "Saturday" , "Sunday"])
for col in X :
   if col =="day":
       X[col] = le1.transform(X[col])       

         
#to only select features that give the best results
X = SelectKBest(mutual_info_regression , k = 3).fit_transform(X , Y)
min_max_scaler = preprocessing.MinMaxScaler()
X = min_max_scaler.fit_transform(X)
#X = preprocessing.scale(X)

X_train , X_test , Y_train ,Y_test = train_test_split(X , Y , test_size = 0.1 , random_state = 2)

#random forest regressor for prediction
reg = RandomForestRegressor()
reg = reg.fit(X_train , Y_train)

# to make predictions 
Z = reg.predict(X_test)

#to display the score
print(reg.score(X_test , Y_test))    
