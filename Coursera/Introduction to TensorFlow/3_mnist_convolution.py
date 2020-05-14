# https://www.youtube.com/playlist?list=PLkDaE6sCZn6Gl29AoE31iwdVwSG-KnDzF

# https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/Course%201%20-%20Part%206%20-%20Lesson%202%20-%20Notebook.ipynb
# includes by layer CONVOLUTION light up
# Visualizing the Convolutions and Pooling

# https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/Course%201%20-%20Part%206%20-%20Lesson%203%20-%20Notebook.ipynb
# manual convolution examples

# https://github.com/lmoroney/dlaicourse/blob/master/Course%201%20-%20Part%206%20-%20Lesson%202%20-%20Notebook.ipynb
# https://github.com/lmoroney/dlaicourse/blob/master/Course%201%20-%20Part%206%20-%20Lesson%203%20-%20Notebook.ipynb

'''
HAVE TO RESHAPE FOR INPUT
'''

class myCallback (tf.keras.callbacks.Callback):
        def on_epoch_end (self, epoch, logs={}):
            if (logs.get('acc')>0.998):
                print ('\n99.8% accuracy achieved. Stopped.')
                self.model.stop_training=True
                
    callback = myCallback ()
    
    # YOUR CODE ENDS HERE

    mnist = tf.keras.datasets.mnist
    (training_images, training_labels), (test_images, test_labels) = mnist.load_data(path=path)
    # YOUR CODE STARTS HERE
    
    training_images=training_images.reshape(60000, 28, 28, 1)
    test_images = test_images.reshape(10000, 28, 28, 1)
    
    training_images = training_images / 255.0
    test_images = test_images / 255.0

    # YOUR CODE ENDS HERE
    
    n = len (set(training_labels))
    c = 64
    m, i, j, ch = training_images.shape
    
    l = [
            tf.keras.layers.Conv2D (64, 3, input_shape=(i, j, ch), activation='relu'),
            tf.keras.layers.MaxPooling2D (2, 2),
            tf.keras.layers.Conv2D (64, 3, activation='relu'),
            tf.keras.layers.MaxPooling2D (2, 2),
            tf.keras.layers.BatchNormalization (),
            tf.keras.layers.Flatten (),
            tf.keras.layers.Dense (128, activation='relu'),
            tf.keras.layers.Dropout (0.20),
            tf.keras.layers.Dense (n, activation='softmax')
        ]
    
    model = tf.keras.models.Sequential(l)

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    # model fitting
    history = model.fit(training_images, training_labels, epochs=20, callbacks=[callback])
