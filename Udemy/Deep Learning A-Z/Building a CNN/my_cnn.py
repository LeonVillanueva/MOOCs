# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 18:09:02 2019

@author: FartherSkies
"""

from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout

classifier = Sequential ()

# convolutional layers

classifier.add (Convolution2D
                (filters=32, kernel_size=(3,3), input_shape= (64,64,3), activation='relu' ))
classifier.add (MaxPooling2D (pool_size=(2,2)))

## classifier.add (Convolution2D (filters=32, kernel_size=(3,3), activation='relu' ))
## classifier.add (MaxPooling2D (pool_size=(2,2)))

classifier.add(Flatten()) # 1-dimensional vector

classifier.add(Dense(units = 128, kernel_initializer = 'uniform', activation = 'relu'))
classifier.add(Dropout(0.1))
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])