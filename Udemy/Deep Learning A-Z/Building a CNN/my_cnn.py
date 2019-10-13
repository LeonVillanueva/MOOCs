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
from keras.preprocessing.image import ImageDataGenerator

classifier = Sequential ()

# convolutional layers

classifier.add (Convolution2D
                (filters=32, kernel_size=(3,3), input_shape= (64,64,3), activation='relu' ))
classifier.add (MaxPooling2D (pool_size=(2,2)))

classifier.add (Convolution2D (filters=32, kernel_size=(3,3), activation='relu' ))
classifier.add (MaxPooling2D (pool_size=(2,2)))

classifier.add(Flatten()) # 1-dimensional vector

classifier.add(Dropout(0.1))
classifier.add(Dense(units = 128, kernel_initializer = 'uniform', activation = 'relu'))
classifier.add(Dropout(0.1))
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        'dataset/training_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

test_generator = test_datagen.flow_from_directory(
        'dataset/test_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

classifier.fit_generator(
        train_generator,
        steps_per_epoch=6400,
        epochs=8,
        validation_data=test_generator,
        validation_steps=1600)

classifier.save('dogcat.h5')

import numpy as np
from keras.preprocessing import image

test_image = image.load_img('dataset/single_prediction/cat_or_dog_2.jpg', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image)
train_generator.class_indices

if result[0][0] == 1:
    prediction = 'dog'
else:
    prediction = 'cat'

print (prediction)