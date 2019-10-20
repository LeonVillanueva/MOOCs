# -*- coding: utf-8 -*-
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