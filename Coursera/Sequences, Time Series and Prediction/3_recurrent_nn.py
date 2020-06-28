'''
# RNNs
# three dimensional input : (batch, timesteps, 1) = (b * t * 1)
# memory cell input : (batch, 1)
# output (on node) : (batch, neurons) -> Y_0 = H_0
# full node output : (batch, timesteps, neurons)

# sequence to vector RNNs : ignore all Y (and H) except Y_last (H_last)
# return_sequences = True, if needs intra-time outputs, else in tensorflow-keras only Y_last

# default activation is tanh


l = [
	 keras.layers.SimpleRNN (20, return_sequences=True, input_shape=[None, 1]), 	# first dimension is batch size, None = Any
	 keras.layers.SimpleRNN (20), 													# will only return the last Y = H
	 keras.layers.Dense (1) 														# final predictive layer
	]
'''



# lambda layers, arbitrary operations for expanding functionality

time_train = time[:split_time]
x_train = series[:split_time]
time_valid = time[split_time:]
x_valid = series[split_time:]



def windowed_dataset(series, window_size, batch_size, shuffle_buffer):
  dataset = tf.data.Dataset.from_tensor_slices(series)
  dataset = dataset.window(window_size + 1, shift=1, drop_remainder=True)
  dataset = dataset.flat_map(lambda window: window.batch(window_size + 1))
  dataset = dataset.shuffle(shuffle_buffer).map(lambda window: (window[:-1], window[-1]))
  dataset = dataset.batch(batch_size).prefetch(1)
  return dataset

window_size = 28																	# arbitrary, four weeks
batch_size = 128
shuffle_buffer_size = len (x_train) / 10 											# get only 10% of the data

dataset = windowed_dataset(x_train, window_size, batch_size=batch_size, shuffle_buffer=shuffle_buffer_size)

l = [
	 keras.layers.Lambda (lambda x: tf.expand_dims (x, axis=-1), input_shape=[None])# expanding dimensions to 3 without changing 																		   window input
	 keras.layers.SimpleRNN (20, return_sequences=True, input_shape=[None, 1]), 	# first dimension is batch size, None = Any
	 keras.layers.SimpleRNN (20), 													# will only return the last Y = H
	 keras.layers.Dense (1), 														# predictive layer
	 keras.layers.Lambda (lambda x: x * 100.0)										# rescaling tanh ouput (-1,1) to (-100,100)
	]

model = keras.models.Sequential (l)
	
# learning rate

lr_schedule = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 1e-8 * 10**(epoch / 20))

optimizer = tf.keras.optimizers.SGD (lr = 1e-8, momentum = 0.9)

model.compile (
			   loss = tf.keras.losses.Huber(),
			   optimizer = optimizer,
			   metrics=['mae']
			  )
			  
history = model.fit (dataset, epochs=100, callbacks=[lr_schedule])

# Huber loss : https://en.wikipedia.org/wiki/Huber_loss
# notebook 1 : https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/TensorFlow%20In%20Practice/Course%204%20-%20S%2BP/S%2BP%20Week%203%20Lesson%202%20-%20RNN.ipynb