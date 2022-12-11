# -*- coding: utf-8 -*-
"""
Created on Sat May 23 19:20:16 2020

@author: timot
"""


from flask import Flask, render_template, request
from Stock_TA_Analyzer import *
from Stock_NB_Analyzer import *
from Merger_Arbitrage import *
from sqlalchemy import create_engine
import pandas as pd
import numpy as np
from CommonFunctions.commonFunctions import parse_input

#%%

ta_db = 'sqlite:///Database/TAData.db';
ta_engine = create_engine(ta_db, echo=False)

nb_db = 'sqlite:///Database/NBData.db';
nb_engine = create_engine(nb_db, echo=False)

nb_tweets_db = 'sqlite:///Database/NBTweets.db';
nb_tweets_engine = create_engine(nb_tweets_db, echo=False)

app = Flask(__name__)

@app.route('/')
@app.route('/home', methods=['GET', 'POST'])
def home():
    return render_template('home.html')

#%% TA Page    

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
        sqlite_connection = ta_engine.connect()
        data.to_sql(name='TAData', con=sqlite_connection, if_exists='replace')
        sqlite_connection.close()
    else:
        sqlite_connection = ta_engine.connect()
        data = pd.read_sql('select * FROM TAData', sqlite_connection)
        sqlite_connection.close()
        ta_analyzer.load_data(data)
    
    macd_p,rsi_p,obv_p,analysis_p = ta_analyzer.compute_ta(slow_ema,fast_ema,signal_ema,rsi_sell_thresh,rsi_buy_thresh,\
                   filter_win_length,filter_polyorder,True)
    analysisJSON = json.dumps(analysis_p, cls=plotly.utils.PlotlyJSONEncoder)
        
    svr_p = ta_analyzer.preprocess_and_train(days_ahead,days_evaluate,train_size,do_smooth,SVM_kernel,SVM_C,SVM_eps,SVM_degree,True)
    svrJSON = json.dumps(svr_p, cls=plotly.utils.PlotlyJSONEncoder)
    
    return render_template('ta_results.html', ticker=ticker, start_date=start_date, recollect_data=recollect_data,\
                           slow_ema=slow_ema, fast_ema=fast_ema, signal_ema=signal_ema, rsi_sell_thresh=rsi_sell_thresh, rsi_buy_thresh=rsi_buy_thresh,\
                               filter_win_length=filter_win_length, filter_polyorder=filter_polyorder, days_ahead=days_ahead, days_evaluate=days_evaluate,\
                                   do_smooth=do_smooth, train_size=train_size, SVM_C=SVM_C, SVM_kernel=SVM_kernel, SVM_degree=SVM_degree,\
                                       SVM_eps=SVM_eps, analysisJSON=analysisJSON, svrJSON=svrJSON)

#%% NB Page

@app.route('/define_nb', methods=['GET', 'POST'])
def define_nb_parameters():
    return render_template('nb_results.html', ticker='ATVI', start_date='01-18-2022', recollect_data=True,\
                               userName='', sinceDate='01-18-2022', untilDate='03-01-2022', querySearch='ATVI AND MSFT',\
                                   maxTweets=10, lang='en', recollectTweets=True, deltaInterval='1,2', changeThreshold='0',\
                                       trainSize='0.8', useIDF=True, do_downsample=True, useStopwords=True, addStopwords='',\
                                           userNamePredict='', querySearchPredict='ATVI AND MSFT', maxTweetsPredict=10, langPredict='en')

@app.route('/nb_results', methods=['GET', 'POST'])
def nb_results():
    
    ticker = request.form['ticker']
    start_date = request.form['start_date']
    recollect_data = bool(int(request.form['recollect_data']))
    
    userName = request.form['userName']
    sinceDate = request.form['sinceDate']
    untilDate = request.form['untilDate']
    querySearch = request.form['querySearch']
    maxTweets = parse_input(request.form['maxTweets'], 10, expect_output='int')[0]
    lang = request.form['lang']
    recollectTweets = bool(int(request.form['recollect_data']))
    
    deltaInterval = parse_input(request.form['deltaInterval'], [1,2], expect_output='int')
    changeThreshold = parse_input(request.form['changeThreshold'], [0,0.02], expect_output='float')
    trainSize = parse_input(request.form['trainSize'], [0.8], expect_output='float')
    useIDF = request.form['userName']
    if useIDF == 'True':
        useIDFActual = [True]
    elif useIDF == 'False':
        useIDFActual = [False]
    else:
        useIDFActual = [True, False]
    do_downsample = request.form['do_downsample']
    if do_downsample == 'True':
        do_downsample_act = [True]
    elif do_downsample == 'False':
        do_downsample_act = [False]
    else:
        do_downsample_act = [True, False]
    useStopwords = bool(int(request.form['useStopwords']))
    if useStopwords:
        addStopwords = parse_input(request.form['addStopwords'], '', expect_output='text')[0]
        stopwordsList = [stopwords.words('english')+[addStopwords]]
        stopwordsList = [stopwords.words('english')]
    elif not useStopwords:
        stopwordsList = []
    else:
        addStopwords = parse_input(request.form['addStopwords'], '', expect_output='text')[0]
        stopwordsList = [stopwords.words('english')+[addStopwords],[]]
    
    userNamePredict = request.form['userNamePredict']
    querySearchPredict = request.form['querySearchPredict']
    langPredict = request.form['langPredict']
    maxTweetsPredict = parse_input(request.form['maxTweetsPredict'], 10, expect_output='int')[0]
    
    NB_analyzer = Stock_NB_Analyzer()
    if recollectTweets:
        NB_analyzer.collect_tweets(userName, sinceDate, untilDate, querySearch, maxTweets, lang)
        tweets = NB_analyzer.tweetsDF_
        sqlite_connection = nb_tweets_engine.connect()
        tweets.to_sql(name='NBTweets', con=sqlite_connection, if_exists='replace')
        sqlite_connection.close()
    else:
        sqlite_connection = nb_tweets_engine.connect()
        tweets = pd.read_sql('select * from NBTweets', sqlite_connection)
        sqlite_connection.close()
        NB_analyzer.load_tweets(tweets)
    
    if recollect_data:
        NB_analyzer.collect_data(ticker, start_date)
        data = NB_analyzer.stockData_
        sqlite_connection = nb_engine.connect()
        data.to_sql(name='NBData', con=sqlite_connection, if_exists='replace')
        sqlite_connection.close()
    else:
        sqlite_connection = nb_engine.connect()
        data = pd.read_sql('select * from NBData', sqlite_connection)
        sqlite_connection.close()
        NB_analyzer.load_data(data)
        
    NB_analyzer, count_report, twitter_plot, report, most_inform, conf_mat, deltaInterval, changeThreshold, trainSize, useIDF, do_downsample, stopwordsList = \
            Stock_NB_Grid_Search(NB_analyzer, trainSize, deltaInterval, changeThreshold, useIDFActual, do_downsample_act, stopwordsList, True, 10, True)
    
    twitter_plot_JSON = json.dumps(twitter_plot, cls=plotly.utils.PlotlyJSONEncoder)
    
    pred_results, pred_text = NB_analyzer.run_prediction(userNamePredict, querySearchPredict, maxTweetsPredict, langPredict, True)
        
    return render_template('nb_results.html', ticker=ticker, start_date=start_date, recollect_data=recollect_data,\
                               userName=userName, sinceDate=sinceDate, untilDate=untilDate, querySearch=querySearch,\
                                   maxTweets=maxTweets, lang=lang, recollectTweets=recollectTweets,\
                                       deltaInterval=str(deltaInterval).replace(' ','').replace('[','').replace(']',''),\
                                           changeThreshold=str(changeThreshold).replace(' ','').replace('[','').replace(']',''),\
                                       trainSize=str(trainSize).replace(' ','').replace('[','').replace(']',''), useIDF=useIDF,\
                                           do_downsample=do_downsample, useStopwords=useStopwords, addStopwords='',\
                                           userNamePredict=userNamePredict, querySearchPredict=querySearchPredict, maxTweetsPredict=maxTweetsPredict,\
                                               langPredict=langPredict, twitter_plot=twitter_plot_JSON, count_report=count_report, report=report.to_html(index=False),\
                                                   conf_mat=conf_mat.to_html(index=False), most_inform=most_inform.to_html(index=False), pred_results=pred_results, pred_text=pred_text)
        
#%% Merger Arb Page

merger_analyzer = Merger_Arbitrage()

@app.route('/define_ma', methods=['GET', 'POST'])
def define_ma_parameters():
    return render_template('ma_results.html', analyzeTicker='ATVI', announceDate='01-18-2022',\
                           acqPrice=95, refTicker='ESPO', userName='', querySearch='ATVI AND MSFT',\
                               maxTweets=25, probThresh=2.5, dayGuard=1, lang='en')
        
@app.route('/ma_results', methods=['GET', 'POST'])
def ma_results():
    
    analyzeTicker = request.form['analyzeTicker']
    announceDate = request.form['announceDate']
    acqPrice = parse_input(request.form['acqPrice'], 95, expect_output='float')[0]
    refTicker = request.form['refTicker']
    
    userName = request.form['userName']
    querySearch = request.form['querySearch']
    maxTweets = parse_input(request.form['maxTweets'], 25, expect_output='int')[0]
    lang = request.form['lang']
    probThresh = parse_input(request.form['probThresh'], 2.5, expect_output='float')[0]
    dayGuard = parse_input(request.form['dayGuard'], 1, expect_output='int')[0]
    
    merger_analyzer.collect_data(analyzeTicker,refTicker,announceDate)
    merger_analyzer.find_probability(acqPrice)
    merger_analyzer.collectTweets(querySearch, probThresh, dayGuard, maxTweets, userName, lang)
    mostInform = merger_analyzer.evalDictionStats(doHTML=True)
    fig1, fig2, tweetString = merger_analyzer.plot_data(doHTML=True)
    
    fig1_json = json.dumps(fig1, cls=plotly.utils.PlotlyJSONEncoder)
    fig2_json = json.dumps(fig2, cls=plotly.utils.PlotlyJSONEncoder)
        
    return render_template('ma_results.html', analyzeTicker=analyzeTicker, announceDate=announceDate,\
                           acqPrice=acqPrice, refTicker=refTicker, userName=userName, querySearch=querySearch,\
                               maxTweets=maxTweets, probThresh=probThresh, dayGuard=dayGuard, lang=lang,\
                                   fig1=fig1_json, fig2=fig2_json,\
                                   mostInform=mostInform.to_html(index=False), tweetString=tweetString)
        
@app.route('/ma_requery', methods=['GET', 'POST'])
def ma_requery():
    
    userNameRequery = request.form['userNameRequery']
    requeryDate = request.form['requeryDate']
    requerySearch = request.form['requerySearch']
    requeryNumTweets = parse_input(request.form['requeryNumTweets'], 25, expect_output='int')[0]
    requeryLang = request.form['requeryLang']
    
    requeryString = merger_analyzer.requeryTweets(requerySearch,requeryDate,requeryNumTweets,userNameRequery,requeryLang)
    
    fig1_json = json.dumps(merger_analyzer.fig1_, cls=plotly.utils.PlotlyJSONEncoder)
    fig2_json = json.dumps(merger_analyzer.fig2_, cls=plotly.utils.PlotlyJSONEncoder)
        
    return render_template('ma_results.html', analyzeTicker=merger_analyzer.analyzeTicker_, announceDate=merger_analyzer.announceDate_,\
                           acqPrice=merger_analyzer.acqPrice_, refTicker=merger_analyzer.refTicker_, userName=merger_analyzer.userName_,\
                               querySearch=merger_analyzer.searchString_, maxTweets=merger_analyzer.maxTweets_, probThresh=merger_analyzer.probThresh_,\
                                   dayGuard=merger_analyzer.dayGuard_, lang=merger_analyzer.language_, fig1=fig1_json, fig2=fig2_json,\
                                   mostInform=merger_analyzer.mostInform_.to_html(index=False), tweetString=merger_analyzer.tweetString_,\
                                       userNameRequery=userNameRequery, requeryDate=requeryDate, requerySearch=requerySearch, requeryNumTweets=requeryNumTweets,\
                                           requeryLang=requeryLang, requeryString=requeryString)