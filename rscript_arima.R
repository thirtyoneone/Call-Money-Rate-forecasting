library(readxl)
rates <- read_excel("D:/Dev/4-1/Project - Monetary Policy/Dataset/rates.xlsx")
View(rates)
ratests = ts(rates)
ratests = ts(rev(ratests))
plot(ratests)
library(urca)
library(aTSA)
library(forecast)
library(TTR)
stationary.test(ratests, method = 'pp')
stationary.test(ratests, method = 'kpss')
ratesdiff1 = diff(ratests, differences = 1)
stationary.test(ratesdiff1, method = 'pp')
stationary.test(ratesdiff1, method = 'kpss')
ratesdiff2 = diff(ratests, differences = 2)
stationary.test(ratesdiff2, method = 'pp')
stationary.test(ratesdiff2, method = 'kpss')
acf(ratesdiff2, lag.max = 20)
pacf(ratesdiff2, lag.max = 20)
auto.arima(ratesdiff2)
auto.arima(ratests)
auto.arima(ratesdiff1)
ratesarima = arima(ratests, order = c(0,2,2))
ratesforecast = forecast(ratesarima, h = 5)
plot(ratesforecast)