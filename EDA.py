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

#loading the dataset
#using header = none because the excel file only has data
series = np.array(pd.read_excel("D:\\Dev\\4-1\\Project - Monetary Policy\\Dataset\\rates.xlsx", header = None))
series = np.reshape(series, (series.shape[0],))
series = np.flip(series)
time = np.arange(566)

plot_graph(time,series)
plt.show()

dodchange = list()

for i in range(1, len(series)):
	value = series[i] - series[i - 1]
	dodchange.append(value)

#print(dodchange)

time1 = np.arange(565)
plot_graph(time1, dodchange)
plt.show()
