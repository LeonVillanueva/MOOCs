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
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
    # can skip to OneHot immidiately > LabelEncoder deprecated
    # labelencoder = LabelEncoder()
    # X[:, 3] = labelencoder.fit_transform()
onehotencoder = OneHotEncoder()
X_oh = onehotencoder.fit_transform(X[:,3:4]).toarray()
X_oh = X_oh[:,:-1]
X = np.concatenate ((X[:,:3],X_oh), axis=1)

