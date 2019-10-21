# hybird supervised (deep learning) and unsupervised (SOM)

"""
Created on Sat Oct 19 21:41:29 2019
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
frauds = np.concatenate((mappings[(2,2)], mappings[(5,7)]), axis = 0)
frauds = sc.inverse_transform(frauds)

