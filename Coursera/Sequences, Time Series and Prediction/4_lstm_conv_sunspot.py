def windowed_dataset(series, window_size, batch_size, shuffle_buffer):

  series =  tf.expand_dims (series, axis=-1) # adapting for Conv1D, similar function to lambda layer

  dataset = tf.data.Dataset.from_tensor_slices(series)
  dataset = dataset.window(window_size + 1, shift=1, drop_remainder=True)
  dataset = dataset.flat_map(lambda window: window.batch(window_size + 1))
  dataset = dataset.shuffle(shuffle_buffer).map(lambda window: (window[:-1], window[-1]))
  dataset = dataset.batch(batch_size).prefetch(1)
  return dataset

l = [
	 # keras.layers.Lambda (lambda x: tf.expand_dims (x, axis=-1), input_shape=[None])
	 tf.keras.Conv1D (
					filters = 32,
					kernel_size = 5,
					strides = 1,
					padding = 'causal',
					activation = 'relu',
					input_shape = [None, 1]
				  ),
	 tf.keras.LSTM(32), return_sequence=True),			
	 tf.keras.LSTM(32),								
	 tf.keras.layers.Dense (1), 														
	 tf.keras.layers.Lambda (lambda x: x * 200.0)										
	]

model = tf.keras.models.Sequential (l)

optimizer = tf.keras.optimizer.SGD (lr=1e-5, momentum=0.9)

model.compile (
			   loss = tf.keras.losses.Huber(),
			   optimizer = optimizer,
			   metrics=['mae']
			  )

history = model.fit (dataset, epoch=500)

# notebook1 : https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/TensorFlow%20In%20Practice/Course%204%20-%20S%2BP/S%2BP%20Week%204%20Lesson%201.ipynb

''' # forecast helper function

def model_forecast(model, series, window_size):
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(window_size, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(window_size))
    ds = ds.batch(32).prefetch(1)
    forecast = model.predict(ds)
    return forecast

'''

# notebook2 : https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/TensorFlow%20In%20Practice/Course%204%20-%20S%2BP/S%2BP%20Week%204%20Lesson%201.ipynb#scrollTo=_eaAX9g_jS5W https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/TensorFlow%20In%20Practice/Course%204%20-%20S%2BP/S%2BP%20Week%204%20Lesson%205.ipynb https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/TensorFlow%20In%20Practice/Course%204%20-%20S%2BP/S%2BP%20Week%204%20Lesson%203.ipynb


'''
window size


'''

# notebook3 : sunspots : https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/TensorFlow%20In%20Practice/Course%204%20-%20S%2BP/S%2BP%20Week%204%20Exercise%20Answer.ipynb
# notebook3 : sunspots : https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/TensorFlow%20In%20Practice/Course%204%20-%20S%2BP/S%2BP%20Week%204%20Exercise%20Answer.ipynb