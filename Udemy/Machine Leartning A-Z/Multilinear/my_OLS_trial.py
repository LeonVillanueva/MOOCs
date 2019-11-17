# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 19:01:59 2019

@author: FartherSkies
"""

import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.sandbox.regression.predstd import wls_prediction_std

np.random.seed(9876789)

nsample = 100
x = np.linspace(0, 10, 100)
X = np.column_stack((x, x**2))
beta = np.array([1, 0.1, 10])
e = np.random.normal(size=nsample)

X = sm.add_constant(X)
y = np.dot(X, beta) + e

print (X.shape, y.shape)
print (X.dtype, y.dtype)

model = sm.OLS(y, X)
results = model.fit()
print(results.summary())