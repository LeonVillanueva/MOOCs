# -*- coding: utf-8 -*-
"""
Created on Sun Nov 10 09:30:19 2019

@author: FartherSkies
"""

''' template '''
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values # preserve the 2D array
y = dataset.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)


from sklearn.linear_model import LinearRegression
lr = LinearRegression ()

lr.fit (X_train, y_train)
y_pred = lr.predict (X_test)

lr.coef_
lr.intercept_

import matplotlib.pyplot as plt

