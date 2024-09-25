---
title: "Sales Price Prediction Model"
author: "Ankit"
date: "2023-05-01"
output: html_document
---

## Overview

This project uses various time series forecasting models to predict U.S. retail sales data. The dataset contains monthly sales data (in millions of dollars) starting from 1992. In this project, we explore several forecasting techniques including seasonal naive, ETS, ARIMA, and a neural network-based approach.

## Data Loading

We start by loading the sales data from a CSV file:

```{r}
# Loading the initial data
data <- read.csv("/Users/ankitdas/Downloads/Sales.csv")
# Converting data into time series
Y <- ts(data[,2], start = c(1992,1), frequency = 12)
```
## A simple plot of the raw sales data
```{r}
autoplot(Y) + ggtitle("Time Plot: US Retail Sales per Month") + ylab("Millions of Dollars")
```
## To remove the trend in the data, we apply first-order differencing:
```{r}
DY <- diff(Y)
autoplot(DY) + ggtitle("Time Plot: Change in US Retail Sales per Month") + ylab("Millions of Dollars")
```

## To remove the trend in the data, we apply first-order differencing:
```{r}
DY <- diff(Y)
autoplot(DY) + ggtitle("Time Plot: Change in US Retail Sales per Month") + ylab("Millions of Dollars")
```
We also analyze the autocorrelation (ACF) and partial autocorrelation (PACF) to understand the structure of the data:
```{r}
acf(DY)
pacf(DY)
```

## Seasonality Analysis
We check the seasonality of the data using season plots and subseries plots:
```{r}
ggseasonplot(DY) + ggtitle("Season Plot: Change in US Retail Sales per Month") + ylab("Millions of Dollars")
ggsubseriesplot(DY)

```
## Forecasting Models
Seasonal Naive Model
```{r}
fit <- snaive(DY)
summary(fit)
checkresiduals(fit)

```
ETS (Exponential Smoothing State Space) Model
```{r}
fit_ets <- ets(Y)
summary(fit_ets)
checkresiduals(fit_ets)

```
 ARIMA Model
 ```{r}
fit_arima <- auto.arima(Y, d=1, D=1, stepwise = FALSE, approximation = FALSE)
summary(fit_arima)
checkresiduals(fit_arima)

fcst_arima <- forecast(fit_arima, h=12)
autoplot(fcst_arima)

```
Neural Network Forecasting
```{r}
train_data <- window(Y, end=c(2018,12))
test_data <- window(Y, start=c(2019,1))

# Neural network model
model_nnetar <- nnetar(train_data)

# Forecast for the next 12 months
forecast_nnetar <- forecast(model_nnetar, h=12)

# Plotting the results
plot(Y, type="l", col="blue", main="Neural Network Forecast vs. Actual Sales", xlab="Year", ylab="Sales (millions of dollars)")
lines(fitted(forecast_nnetar), col="red")
lines(forecast_nnetar$mean, col="green")
legend("topleft", legend=c("Actual Sales", "Fitted Sales", "Forecasted Sales"), col=c("blue", "red", "green"), lty=1)

```
## Conclusion
This project explored several forecasting techniques to model U.S. retail sales. The ARIMA and ETS models provided solid results,
while the neural network model gave a modern, flexible approach to the task. Each model offers distinct insights depending on the
assumptions and data structures, and the choice of the best model depends on specific forecasting goals.

