# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 18:52:27 2021

@author: timot
"""


from Stock_TA_Analyzer import *

#%% Inputs

# Stock Data
start_date = '06-01-2020' # %m-%d-%Y
ticker = 'QQQ'

# MACD
slow_ema = 26 #26 default
fast_ema = 12 #12 default
signal_ema = 9 #9 default

# RSI
rsi_sell_thresh = 70
rsi_buy_thresh = 50

# Smoothing Filter Parameters
filter_win_length = 5
filter_polyorder = 1

# ML Parameters
days_ahead = 4 #predict days ahead (>=1)
days_evaluate = 3 #past days for input
train_size = 0.5
do_smooth = True

# SVR Parameters
kernel = 'rbf'
C = 0.1
eps = 1e-5
degree = 3

doHTML = False

#%% Execute

ta_analyzer = Stock_TA_Analyzer()
ta_analyzer.collect_data(ticker,start_date)
ta_analyzer.compute_ta(slow_ema,fast_ema,signal_ema,rsi_sell_thresh,rsi_buy_thresh,\
                   filter_win_length,filter_polyorder,doHTML)
ta_analyzer.preprocess_and_train(days_ahead,days_evaluate,train_size,do_smooth,kernel,C,eps,degree,doHTML)