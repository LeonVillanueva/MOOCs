# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 18:15:35 2019

@author: FartherSkies
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')
training_set = dataset_train.iloc[:, 1:2].values

# print (training_set[:10])

# scaling
# normalization = sigmoid
# standardization = negatives, regression

from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler (feature_range=(0,1))
training_set_norm = sc.fit_transform(training_set)

# observation of 60 day lag

X_train = []
y_train = []

print (len(training_set_norm))

for i in range (60, len(training_set_norm)):
    X_train.append (training_set_norm[i-60:i,0])
    y_train.append (training_set_norm[i, 0])