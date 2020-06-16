# time series

# forecasts, inputation, anomaly detection, pattern recognition (sound for speach recognition)
# trend, seasonality, autocorrelation, non-stationary time series (event driven systemic changes)

# notebook1 : https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/TensorFlow%20In%20Practice/Course%204%20-%20S%2BP/S%2BP_Week_1_Lesson_2.ipynb

# fixed partitioning, capture all seasons
# roll forward partitioning

# errors = forecast - actual = yhat - yhat
# mse = np.sqrt(errors).mean()
# rmse = np.sqrt(mse)
# mae = np.abs(errors).mean()
# mape = = np.abs(errors / x_valid).mean()

# moving average and differencing : no trend / no seasonality
# remove trend and seasonality : x(t) - x(t-1)

# trailing versus centered window
# notebook2 : https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/TensorFlow%20In%20Practice/Course%204%20-%20S%2BP/S%2BP%20Week%201%20-%20Lesson%203%20-%20Notebook.ipynb

# notebook3 : https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/TensorFlow%20In%20Practice/Course%204%20-%20S%2BP/Week%201%20Exercise%20Question.ipynb