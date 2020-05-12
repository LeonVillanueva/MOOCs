# callback function for loss 
# https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/Course 1 - Part 4 - Lesson 4 - Notebook.ipynb

class myCallback (tf.keras.callbacks.Callback):
	def on_epoch_end (self, epoch, logs={}):
		if (logs.get('loss')<0.4):
			print ('\nLoss low, training suspended.')
			self.model.stop_training=True

callback = myCallback ()

fashion_mnist = keras.datasets.fashion_mnist
(train, train_labels), (test, test_labels) = fashion_mnist.load_data ()

# https://developers.google.com/machine-learning/fairness-overview/

train = train / np.max (train)
test = test / np.max (train)

n = 10
l = [
	 tf.keras.layers.Flatten (input_shape=(28,28)),
	 tf.keras.layers.Dense (128, activation='relu'),
	 tf.keras.layers.Dropout (0.25),
	 tf.keras.layers.Dense (n, activation='softmax')
	]
model = tf.keras.seqential(l)

model.compile (optimizer='adam',
               loss='sparse_categorical_crossentropy',
               metrics=['accuracy'])

model.fit (train, train_labels, epochs=100, callbacks=[callback])

'''
final
'''

class myCallback (tf.keras.callbacks.Callback):
        def on_epoch_end (self, epoch, logs={}):
            if (logs.get('acc')>0.99):
                print ('\nReached 99% accuracy so cancelling training!')
                self.model.stop_training=True

    callback = myCallback ()

    # YOUR CODE SHOULD END HERE
    mnist = tf.keras.datasets.mnist

    (x_train, y_train),(x_test, y_test) = mnist.load_data(path=path)
    x_train = x_train / 255.0
    x_test = x_test / 255.0
    
    n = 10
    
    l = [tf.keras.layers.Flatten (input_shape=(28,28)),
         tf.keras.layers.Dense (128*4, activation='relu'),
         tf.keras.layers.Dropout (0.25),
         tf.keras.layers.Dense (n, activation='softmax')]


    # YOUR CODE SHOULD END HERE
    model = tf.keras.models.Sequential(l)

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])


