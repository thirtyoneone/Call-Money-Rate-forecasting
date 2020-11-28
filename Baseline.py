import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#creating a function to plot the graph for given series and time period
def plot_graph(time, series):
	plt.figure(figsize =(10,6))
	plt.plot(time,series)
	plt.xlabel("time period")
	plt.ylabel("data value")
	#plt.grid(True)

def error(valid,forecast):
	print(keras.metrics.mean_squared_error(valid, forecast).numpy())
	print(keras.metrics.mean_absolute_error(valid, forecast).numpy())

#loading the dataset
#using header = none because the excel file only has data
series = np.array(pd.read_excel("D:\\Dev\\4-1\\Project - Monetary Policy\\Dataset\\rates.xlsx", header = None))
series = np.reshape(series, (series.shape[0],))
series = np.flip(series)
time = np.arange(566)

plot_graph(time,series)
plt.show()

#setting variables
split = 350
time_train = time[:split]
series_train = series[:split]
time_valid = time[split:]
series_valid = series[split:]

naive_forecast = series[split - 1: -1]

error(series_valid, naive_forecast)

