# hybird supervised (deep learning) and unsupervised (SOM)

"""
Created on Sat Oct 22 21:41:29 2019
@author: FartherSkies
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv ('Credit_Card_Applications.csv')

# mean interneuron distance
# euclidean distance between neuron - within a (defined) neighborhood

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# feature scaling

from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
X_norm = sc.fit_transform (X)

# miniSOM
from minisom import MiniSom
som = MiniSom (x=10, y=10, input_len=X_norm.shape[1])

# weight initialization
som.random_weights_init(X_norm)
som.train_random (X_norm, 100)

from pylab import bone, pcolor, colorbar, plot, show
bone()
pcolor(som.distance_map().T)
colorbar()
markers = ['3', 'o']
colors = ['r', 'g']

for i, x in enumerate (X_norm):
    w = som.winner (x)
    plot(w[0]+0.5,w[1]+0.5, markers[y[i]], markeredgecolor=colors[y[i]], 
         markerfacecolor='None', markersize=8, markeredgewidth=1)
        # +0.5 centering
        
show()

mappings = som.win_map(X_norm)
frauds = mappings[(8,4)]
frauds = sc.inverse_transform(frauds)

frauds_list = frauds[:,0]
len (frauds_list)
# create a dependent variable from the SOM

customers = dataset.iloc[:, 1:].values
is_fraud = np.zeros (customers.shape[0])

for i in range (customers.shape[0]):
    if dataset.iloc[i,0] in frauds_list:
        is_fraud[i] = 1.


'''
from ANN.py
'''

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler ()
X_train = sc.fit_transform(customers)

# Importing the Keras libraries and packages
import keras
from keras.models import Sequential # required for initialization
from keras.layers import Dense

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(units = 3, kernel_initializer = 'uniform', activation = 'relu', input_dim = X_train.shape[1]))

# Adding the output layer
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
 
'''   
https://towardsdatascience.com/advanced-keras-constructing-complex-custom-losses-and-metrics-c07ca130a618
'''

# Fitting the ANN to the Training set
classifier.fit(X_train, is_fraud, batch_size = 2, epochs = 3)

# Predicting the Test set results
y_pred = classifier.predict(customers)

y_pred = np.concatenate(dataset.iloc[:,0:1], y_pred, axis=1) # make into the same array dimension -> 2D array
y_pred = y_pred[y_pred[:,1].argsort()]
