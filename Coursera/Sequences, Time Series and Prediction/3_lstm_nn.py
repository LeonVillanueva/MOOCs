'''
# cell state is kept, so the effect is less diminished over "time"
# cell state can bidirectional

# https://www.coursera.org/lecture/nlp-sequence-models/long-short-term-memory-lstm-KXoay
'''

# notebook2: https://www.coursera.org/learn/tensorflow-sequences-time-series-and-prediction/supplement/EtewX/lstm-notebook

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

tf.keras.backend.clear_session()

l = [
	 keras.layers.Lambda (lambda x: tf.expand_dims (x, axis=-1), input_shape=[None])# expanding to three dimensions
	 keras.layers.Bidirectional (tf.keras.LSTM(32), return_sequence=True),			# have to always return_sequences for anything 
																					# but last
	 keras.layers.Bidirectional (tf.keras.LSTM(32)),								
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
			   metrics=['mse']
			  )
			  
history = model.fit (dataset, epochs=100, callbacks=[lr_schedule])
# model.fit (dataset, epochs=100, callbacks=[lr_schedule], verbose=0)