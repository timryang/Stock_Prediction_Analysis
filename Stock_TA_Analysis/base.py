# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 18:52:27 2021

@author: timot
"""


from CommonFunctions.commonFunctions import *
import ta
import numpy as np
import pandas as pd
from datetime import date
from matplotlib import pyplot as plt
from scipy.signal import savgol_filter
from sklearn import svm

#%% Inputs

today = date.today()

# Stock Data
start_date = '06-01-2020' # %m-%d-%Y
end_date = today.strftime('%m-%d-%Y') # %m-%d-%Y
ticker = 'QQQ'

# MACD
slow_ema = 26 #26 default
fast_ema = 12 #12 default
signal_ema = 9 #9 default

# RSI
rsi_sell_thresh = 70
rsi_buy_thresh = 50

# Smoothing Filter Parameters
window_length = 5
polyorder = 1

# ML Parameters
days_ahead = 4 #predict days ahead (>=1)
days_evaluate = 3 #past days for input
train_size = 0.5
do_smooth = True

# SVR Parameters
kernel = 'rbf'
C = 0.1
eps = 1e-5

#%% Collect Data

data = collect_stock_data(ticker, start_date, end_date)
close_data = data['Close']
dates = np.array(pd.to_datetime(data['Date']))

#%% Compute TA

macd_obj = ta.trend.MACD(close_data,slow_ema,fast_ema,signal_ema)
obv_obj = ta.volume.on_balance_volume(close_data, data['Volume'])
rsi_obj = ta.momentum.rsi(close_data)

# RSI lines
x_date_line = [dates[0],dates[-1]]
rsi_sell_line = rsi_sell_thresh*np.ones(2)
rsi_buy_line = rsi_buy_thresh*np.ones(2)

# Smooth lines
close_data = np.array(close_data)
close_smooth = savgol_filter(close_data, window_length, polyorder)
obv_smooth = savgol_filter(obv_obj, window_length, polyorder)

#%% Compute MACD Crossovers

macd_buy = dates[np.where(np.sign(np.array(macd_obj.macd_diff()[:-1]))<np.sign(np.array(macd_obj.macd_diff()[1:])))[0]]
macd_sell = dates[np.where(np.sign(np.array(macd_obj.macd_diff()[:-1]))>np.sign(np.array(macd_obj.macd_diff()[1:])))[0]]
macd_buy = list(np.tile(macd_buy,(2,1)).T)
macd_sell = list(np.tile(macd_sell,(2,1)).T)
vert_line = np.array([np.min(close_data),np.max(close_data)])

#%% Compute RSI Regions

rsi_buy = dates[np.where(rsi_obj<rsi_buy_thresh)[0]]
rsi_sell = dates[np.where(rsi_obj>rsi_sell_thresh)[0]]
vert_point = np.min(close_data)-0.01*np.min(close_data)

#%% Compute OBV Regions

# Possibly compare slopes between obv and close to find divergence
obv_slope = np.diff(obv_smooth)
obv_slope /= max(obv_slope)
obv_slope_smooth = savgol_filter(obv_slope, window_length, polyorder)

#%% Plot

plot_values([dates,dates], [macd_obj.macd(),macd_obj.macd_signal()], ['MACD','Signal'], 'Date', 'MACD', 'MACD', True)
plot_values([dates,x_date_line,x_date_line], [rsi_obj,rsi_sell_line,rsi_buy_line], ['RSI','Sell Thresh','Buy Thresh'], 'Date', 'RSI', 'RSI', True)
plot_values([dates,dates], [obv_obj,obv_smooth], ['OBV Raw','OBV Smooth'], 'Date', 'OBV', 'OBV', True)

plt1_x = [dates,dates]+macd_buy+macd_sell+[rsi_buy,rsi_sell]
plt1_y = [close_data,close_smooth]+[vert_line]*(len(macd_buy)+len(macd_sell))\
    +[np.tile(vert_point,len(rsi_buy)),np.tile(vert_point,len(rsi_sell))]
plt1_labels = ['Close Raw','Close Smooth']+['MACD Buy']*len(macd_buy)+['MACD Sell']*len(macd_sell)\
    +['RSI Buy','RSI Sell']
plt1_lines = ['C0-','C1-']+['g--']*len(macd_buy)+['r--']*len(macd_sell)+['go','ro']
plot_values(plt1_x, plt1_y, plt1_labels, 'Date', 'Close', 'Analysis', True, False, plt1_lines)

#%% Preprocess for ML

# Consider smooth  data

# Form data
close_diff_perc_smooth = (close_smooth[days_ahead:]-close_smooth[:-days_ahead])/close_smooth[:-days_ahead] #percent change
close_diff_perc = (close_data[days_ahead:]-close_data[:-days_ahead])/close_data[:-days_ahead] #percent change
macd_diff = macd_obj.macd_diff()

# Standardize
macd_diff_std = (macd_diff-np.mean(macd_diff))/np.std(macd_diff)
rsi_std = (rsi_obj-np.mean(rsi_obj))/np.std(rsi_obj)
if do_smooth:
    obv_slope_std =  (obv_slope_smooth-np.mean(obv_slope_smooth))/np.std(obv_slope_smooth)
else:
    obv_slope_std =  (obv_slope-np.mean(obv_slope))/np.std(obv_slope)

# Create input/output matrix
first_idx = np.max([macd_diff.first_valid_index(),rsi_obj.first_valid_index()]) #first valid idx
m_size = macd_diff.size-(first_idx+days_evaluate+days_ahead-1) #rows (days)
n_size = days_evaluate*3 #columns (n_features)
if do_smooth:
    output = close_diff_perc_smooth[first_idx+days_evaluate-1:]
else:
    output = close_diff_perc[first_idx+days_evaluate-1:]
input_array = np.zeros((m_size,n_size))
for i_m in range(m_size):
    start_idx = first_idx+i_m
    input_array[i_m,:] = np.r_[macd_diff_std[start_idx:start_idx+days_evaluate],\
                               rsi_std[start_idx:start_idx+days_evaluate],\
                                   obv_slope_std[start_idx-1:start_idx-1+days_evaluate]]

x_train = input_array[:int(m_size*train_size),:]
y_train = output[:int(m_size*train_size)]
x_test = input_array[int(m_size*train_size):,:]
y_test = output[int(m_size*train_size):]

predict_input = np.zeros((days_ahead,n_size))
for i_m in range(days_ahead):
    start_idx = macd_diff_std.size-(days_ahead+days_evaluate-1)+i_m
    predict_input[i_m,:] = np.r_[macd_diff_std[start_idx:start_idx+days_evaluate],\
                                  rsi_std[start_idx:start_idx+days_evaluate],\
                                      obv_slope_std[start_idx-1:start_idx-1+days_evaluate]]

#%% SVR

clf = svm.SVR(kernel=kernel, C=C, epsilon=eps)
clf.fit(x_train,y_train)
train_output = clf.predict(x_train)
test_output = clf.predict(x_test)
predict_output = clf.predict(predict_input)

#%% Plot

actual = close_diff_perc[first_idx+days_evaluate-1:]
dates_diff = dates[first_idx+days_evaluate+days_ahead-1:]
dates_train = dates_diff[:int(m_size*train_size)]
dates_test = dates_diff[int(m_size*train_size):]
x_values = [dates_diff,dates_diff,dates_train,dates_test]
y_values = [actual,output,train_output,test_output]
labels = ['Actual','Output','Train','Test']
title = 'SVR Predict: '+str(np.round(predict_output,3))
plot_values(x_values, y_values, labels, 'Date', 'Percent Change', title,\
            isDates=True, default_lines=False, lines=['C0--','C1-','C3-','C3--'], isBokeh=False)