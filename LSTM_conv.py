import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

#creating a function to plot the graph for given series and time period
def plot_graph(time, series):
	plt.figure(figsize =(10,6))
	plt.plot(time,series)
	plt.xlabel("time period")
	plt.ylabel("data value")
	#plt.grid(True)

#loading the dataset
#using header = none because the excel file only has data
series = np.array(pd.read_excel("D:\\Dev\\4-1\\Project - Monetary Policy\\Dataset\\rates.xlsx", header = None))
series = np.reshape(series, (series.shape[0],))
#we have 566 entries
time = np.arange(566)
 
#graphing the data
plot_graph(time, series)
plt.show()

#Training and validation partition
split = 450
time_train = time[:split]
time_valid = time[split:]
series_train = series[:split]
series_valid = series[split:]


#setting values
window_size = 25
batch_size = 50
shuffle_buffer_size = 75
ep = 100 #epochs

#error
def error(valid,forecast):
	print(keras.metrics.mean_squared_error(valid, forecast).numpy())
	print(keras.metrics.mean_absolute_error(valid,forecast).numpy())

#creating windowed dataset
def windowed_dataset(series, window_size, batch_size, shuffle_buffer):
  dataset = tf.data.Dataset.from_tensor_slices(series)
  dataset = dataset.window(window_size + 1, shift=1, drop_remainder=True)
  dataset = dataset.flat_map(lambda window: window.batch(window_size + 1))
  dataset = dataset.shuffle(shuffle_buffer).map(lambda window: (window[:-1], window[-1]))
  dataset = dataset.batch(batch_size).prefetch(1)
  return dataset

dataset = windowed_dataset(series_train, window_size, batch_size, shuffle_buffer_size)
print(dataset)

#definig model
model = tf.keras.models.Sequential([
   tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=-1),
                      input_shape=[None]),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, return_sequences=True)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
    tf.keras.layers.Dense(10, activation = "relu"), 
    tf.keras.layers.Dense(1)
])

model.compile(loss="mse", optimizer=tf.keras.optimizers.SGD(lr=1e-6, momentum=0.9))
model.fit(dataset,epochs=ep,verbose=0)

#forecasting
forecast = []
for time in range(len(series) - window_size):
  forecast.append(model.predict(series[time:time + window_size][np.newaxis]))

forecast = forecast[split-window_size:]
results = np.array(forecast)[:, 0, 0]

error(series_valid,results)
