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

print (training_set[:10])
