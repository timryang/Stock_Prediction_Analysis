# -*- coding: utf-8 -*-
"""
Created on Sat May 23 19:20:16 2020

@author: timot
"""


from flask import Flask, render_template, request
from bokeh.embed import components
from Stock_TA_Analyzer import *
from sqlalchemy import create_engine
import pandas as pd
import numpy as np
import bokeh
from CommonFunctions.commonFunctions import parse_input

#%%

database_URI = 'sqlite:///Database/Data.db';
engine = create_engine(database_URI, echo=False)

app = Flask(__name__)

@app.route('/')
@app.route('/home', methods=['GET', 'POST'])
def home():
    return render_template('home.html')
        
@app.route('/define_ta', methods=['GET', 'POST'])
def define_dependency_parameters():
    return render_template('ta_results.html', ticker='QQQ', start_date='05-01-2020', recollect_data=True,\
                           slow_ema=26, fast_ema=12, signal_ema=9, rsi_sell_thresh=70, rsi_buy_thresh=50,\
                               filter_win_length=5, filter_polyorder=1, days_ahead=4, days_evaluate=3,\
                                   do_smooth=True, train_size=0.5, SVM_C=0.1, SVM_kernel='rbf', SVM_degree=3,\
                                       SVM_eps=1e-5)

@app.route('/ta_results', methods=['GET', 'POST'])
def ta_results():
    
    ticker = request.form['ticker']
    start_date = request.form['start_date']
    recollect_data = bool(int(request.form['recollect_data']))
    
    slow_ema = parse_input(request.form['slow_ema'], 26, expect_output='int')[0]
    fast_ema = parse_input(request.form['fast_ema'], 12, expect_output='int')[0]
    signal_ema = parse_input(request.form['signal_ema'], 9, expect_output='int')[0]
    rsi_sell_thresh = parse_input(request.form['rsi_sell_thresh'], 70, expect_output='int')[0]
    rsi_buy_thresh = parse_input(request.form['rsi_buy_thresh'], 50, expect_output='int')[0]
    
    filter_win_length = parse_input(request.form['filter_win_length'], 5, expect_output='int')[0]
    filter_polyorder = parse_input(request.form['filter_polyorder'], 1, expect_output='int')[0]
    
    days_ahead = parse_input(request.form['days_ahead'], 4, expect_output='int')[0]
    days_evaluate = parse_input(request.form['days_evaluate'], 3, expect_output='int')[0]
    do_smooth = bool(int(request.form['do_smooth']))
    train_size = parse_input(request.form['train_size'], 0.5, expect_output='float')[0]
    
    SVM_C = parse_input(request.form['SVM_C'], 0.1, expect_output='float')[0]
    SVM_kernel = (request.form['SVM_kernel']).split(',')[0]
    SVM_degree = parse_input(request.form['SVM_degree'], 3, expect_output='int')[0]
    SVM_eps = parse_input(request.form['SVM_eps'], 1e-5, expect_output='float')[0]
    
    ta_analyzer = Stock_TA_Analyzer()
    if recollect_data:
        ta_analyzer.collect_data(ticker, start_date)
        data = ta_analyzer.stock_data_
        sqlite_connection = engine.connect()
        data.to_sql(name='Data', con=sqlite_connection, if_exists='replace')
        sqlite_connection.close()
    else:
        sqlite_connection = engine.connect()
        data = pd.read_sql('select * from Data', sqlite_connection)
        sqlite_connection.close()
        ta_analyzer.load_data(data)
    
    macd_p,rsi_p,obv_p,analysis_p = ta_analyzer.compute_ta(slow_ema,fast_ema,signal_ema,rsi_sell_thresh,rsi_buy_thresh,\
                   filter_win_length,filter_polyorder,True)
    
    svr_p = ta_analyzer.preprocess_and_train(days_ahead,days_evaluate,train_size,do_smooth,SVM_kernel,SVM_C,SVM_eps,SVM_degree,True)
    
    p = bokeh.layouts.row(analysis_p, svr_p, sizing_mode='stretch_both')
    script_p, div_p = components(p)
    
    # script_an_p, div_an_p = components(analysis_p)
    # script_svr_p, div_svr_p = components(svr_p)
    
    return render_template('ta_results.html', ticker=ticker, start_date=start_date, recollect_data=recollect_data,\
                           slow_ema=slow_ema, fast_ema=fast_ema, signal_ema=signal_ema, rsi_sell_thresh=rsi_sell_thresh, rsi_buy_thresh=rsi_buy_thresh,\
                               filter_win_length=filter_win_length, filter_polyorder=filter_polyorder, days_ahead=days_ahead, days_evaluate=days_evaluate,\
                                   do_smooth=do_smooth, train_size=train_size, SVM_C=SVM_C, SVM_kernel=SVM_kernel, SVM_degree=SVM_degree,\
                                       SVM_eps=SVM_eps, script_p=script_p, div_p=div_p)