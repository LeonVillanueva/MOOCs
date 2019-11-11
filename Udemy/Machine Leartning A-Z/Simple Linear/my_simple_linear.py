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
from sklearn.metrics import r2_score

lr = LinearRegression ()

lr.fit (X_train, y_train)
y_pred = lr.predict (X_test)
r2 = r2_score(y_true, y_pred)

print ('r-squared: '+r2)
print ('coefficent of age: '+lr.coef_)
print ('starting salary: '+lr.intercept_)

loss = np.sqrt ((y_test - y_pred)**2)

# graphical

plt.scatter (X_test, y_test, color='red')
plt.plot (X_test, y_pred, color='blue')
plt.plot (X_test, loss, color='red', alpha=0.30)
plt.title ('Years v Salary')
plt.xlabel ('Years of Experience')
plt.ylabel ('Salary')
plt.show()


