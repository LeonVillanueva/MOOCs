# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 11:35:11 2019

@author: FartherSkies
"""

# Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values