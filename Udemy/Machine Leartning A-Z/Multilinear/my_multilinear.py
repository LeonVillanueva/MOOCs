# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 05:14:58 2019

@author: FartherSkies
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

# Encoding categorical data
from sklearn.preprocessing import OneHotEncoder #, LabelEncoder
    # can skip to OneHot immidiately > LabelEncoder deprecated
    # labelencoder = LabelEncoder()
    # X[:, 3] = labelencoder.fit_transform()
onehotencoder = OneHotEncoder()
X_oh = onehotencoder.fit_transform(X[:,3:4]).toarray()
X_oh = X_oh[:,:-1]
# switched to match
X = np.concatenate ((X_oh, X[:,:3]), axis=1)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)

print (y_pred)
print (regressor.coef_)

# p-vales
# https://www.ritchieng.com/machine-learning-evaluate-linear-regression-model/
# https://stackoverflow.com/questions/22306341/python-sklearn-how-to-calculate-p-values
# https://scikit-learn.org/stable/modules/classes.html#module-sklearn.feature_selection

from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

print (r2_score(y_test, y_pred))
print (mean_squared_error(y_test, y_pred))

import statsmodels.api as sm
# import statsmodels.regression.linear_model as lm

ones = np.ones ((X.shape[0],1), dtype=float)
X_sm = np.append (ones, X, axis=1).astype('float64') 
X_opt = X_sm[:,[0,1,2,3,4,5]]


print (X_opt.shape, y.shape)
print (X_opt.dtype, y.dtype)

model = sm.OLS(y, X_opt, hasconst=True)
results = model.fit()

print (max(results.pvalues))
np.argmax(results.pvalues)

# automatic backstep

X_vars = [0,1,2,3,4,5]
sl = 0.1

while max (results.pvalues) > sl:
    X_vars.pop (np.argmax(results.pvalues))
    model = sm.OLS(y, X_sm[:,X_vars], hasconst=True)
    results = model.fit()

print (X_vars)
print(results.summary())