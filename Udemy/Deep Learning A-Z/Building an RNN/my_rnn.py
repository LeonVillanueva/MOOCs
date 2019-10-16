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
    
X_train, y_train = np.array (X_train), np.array (y_train)

n_id = 1

X_train = np.reshape (X_train, (X_train.shape[0], X_train.shape[1], n_id))

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM

regressor = Sequential ()

regressor.add (LSTM (units=50, return_sequences=True, input_shape=(X_train.shape[1],
                                                                   X_train.shape[2])))
regressor.add (Dropout (0.2))
regressor.add (LSTM (units=50, return_sequences=True))
regressor.add (Dropout (0.2))
regressor.add (LSTM (units=50, return_sequences=True))
regressor.add (Dropout (0.2))
regressor.add (LSTM (units=50))
regressor.add (Dropout (0.2))

# regressor.add (Dense(units = 128, kernel_initializer = 'uniform', activation = 'relu'))
# regressor.add (Dropout (0.2))

regressor.add (Dense(units = 1, kernel_initializer = 'uniform'))
regressor.compile (optimizer = 'adam', loss = 'mean_squared_error')

# regressor.summary ()
regressor.fit (X_train, y_train, epochs=100, batch_size=32)

# predictions
dataset_test = pd.read_csv('Google_Stock_Price_Test.csv')
actual_price = dataset_test.iloc[:, 1:2].values



X_test = []
y_train = []

print (len(training_set_norm))

for i in range (60, len(training_set_norm)):
    X_train.append (training_set_norm[i-60:i,0])
    y_train.append (training_set_norm[i, 0])
