# -*- coding: utf-8 -*-
"""
Created on Sat May 23 19:20:16 2020

@author: timot
"""


from flask import Flask, render_template, request
from bokeh.embed import components
from Stock_NB_Analyzer import *
from Stock_Dependency_Analyzer import *
from Stock_Timing_Analyzer import *
from nltk.corpus import stopwords
from sqlalchemy import create_engine
import pandas as pd
import numpy as np
from CommonFunctions.commonFunctions import parse_input

#%%

database_URI = 'sqlite:///Database/Data.db';
engine = create_engine(database_URI, echo=False)

app = Flask(__name__)

@app.route('/')
@app.route('/home', methods=['GET', 'POST'])
def home():
    return render_template('home.html')

@app.route('/define_nb_parameters', methods=['GET', 'POST'])
def define_nb_parameters():
    return render_template('nb_results.html', ticker='MRNA', years=1, userName='',\
                           geoLocation='latitude,longitude', distance='', sinceDate='2020-03-01', untilDate='2020-05-20',\
                               querySearch='MRNA', topTweets=True, maxTweets=500, lang='en',\
                                   deltaInterval=3, changeThreshold=0.02, trainSize=0.8,\
                                       useIDF=True, do_downsample=True, useStopwords=True, addStopwords='',\
                                           recollectTweets=True, recollectData=True,\
                                           userNamePredict='', geoLocationPredict='latitude,longitude', distancePredict='', querySearchPredict='MRNA',\
                                               topTweetsPredict=True, maxTweetsPredict=10, langPredict='en')

@app.route('/nb_results', methods=['GET', 'POST'])
def nb_results():
        
    ticker = request.form['ticker']
    years = float(request.form['years'])
    userName = request.form['userName']
    if (userName == '') or (userName == 'None'):
        userName = None
    geoLocation = request.form['geoLocation']
    if (geoLocation == 'latitude,longitude') or (geoLocation == '') or (geoLocation == 'None'):
        geoLocation = None
    distance = request.form['distance']
    if (distance == '') or (distance == 'None'):
        distance = None
    else:
        distance = distance + 'mi'
    sinceDate = request.form['sinceDate']
    if (sinceDate == '') or (sinceDate == 'None') or (sinceDate == 'yyyy-mm-dd'):
        sinceDate = None
    untilDate = request.form['untilDate']
    if (untilDate == '') or (untilDate == 'None') or (untilDate == 'yyyy-mm-dd'):
        untilDate = None
    querySearch = request.form['querySearch']
    topTweets = bool(int(request.form['topTweets']))
    maxTweets = int(request.form['maxTweets'])
    lang = request.form['lang']
    if (lang == 'None') or (lang == ''):
        lang = None
    
    deltaInterval = parse_input(request.form['deltaInterval'], None, expect_output='int')
    changeThreshold = parse_input(request.form['changeThreshold'], 0, expect_output='float')
    trainSize = parse_input(request.form['trainSize'], 0.8, expect_output='float')
    try:
        useIDF = [bool(int(request.form['useIDF']))]
    except:
        useIDF = [True, False]
    try:
        do_downsample = [bool(int(request.form['do_downsample']))]
    except:
        do_downsample = [True, False]
    try:
        useStopwords = [bool(int(request.form['useStopwords']))]
    except:
        useStopwords = [True, False]
    addStopwords = request.form['addStopwords']
    add_stopwords_tokenized = addStopwords.split(',')
    stopwordsList = []
    for use_sw in useStopwords:
        if use_sw:
            temp = stopwords.words('english')
            if add_stopwords_tokenized:
                for word in add_stopwords_tokenized:
                    temp.append(word)
            stopwordsList.append(list(set(temp)))
        else:
            stopwordsList.append(None)
        
    recollectData = bool(int(request.form['recollectData']))
    recollectTweets = bool(int(request.form['recollectTweets']))
    
    NB_analyzer = Stock_NB_Analyzer()
    if recollectTweets:
        NB_analyzer.collect_tweets(userName, geoLocation, distance, sinceDate, untilDate, querySearch, maxTweets, topTweets, lang)
        tweetsDF = NB_analyzer.tweetsDF_
        sqlite_connection = engine.connect()
        tweetsDF.to_sql(name='Tweets', con=sqlite_connection, if_exists='replace')
        sqlite_connection.close()
    else:
        sqlite_connection = engine.connect()
        tweetsDF = pd.read_sql('select * from Tweets', sqlite_connection)
        sqlite_connection.close()
        NB_analyzer.load_tweets(tweetsDF)
    if recollectData:
        NB_analyzer.collect_data(ticker, years)
        dataDF = NB_analyzer.stockData_
        sqlite_connection = engine.connect()
        dataDF.to_sql(name='Data', con=sqlite_connection, if_exists='replace')
        sqlite_connection.close()
    else:
        sqlite_connection = engine.connect()
        dataDF = pd.read_sql('select * from Data', sqlite_connection)
        sqlite_connection.close()
        NB_analyzer.load_data(dataDF)
    
    NB_analyzer, count_report, p, report, most_inform, conf_mat, deltaInterval, changeThreshold, trainSize, useIDF, do_downsample, stopwordsList = \
        Stock_NB_Grid_Search(NB_analyzer, trainSize, deltaInterval, changeThreshold, useIDF, do_downsample, stopwordsList)
        
    if stopwordsList == None:
        useStopwords = False
    else:
        useStopwords = True
    
    count_report = count_report.replace('\n', '<br/>')
    script_p, div_p = components(p)
    report = report.to_html(index=False)
    most_inform = most_inform.to_html(index=False)
    conf_mat = conf_mat.to_html(index=False, bold_rows=True)
    
    userNamePredict = request.form['userNamePredict']
    if (userNamePredict == '') or (userNamePredict == 'None'):
        userNamePredict = None
    geoLocationPredict = request.form['geoLocationPredict']
    if (geoLocationPredict == 'latitude,longitude') or (geoLocationPredict == '') or (geoLocationPredict == 'None'):
        geoLocationPredict = None
    distancePredict = request.form['distancePredict']
    if (distancePredict == '') or (distancePredict == 'None'):
        distancePredict = None
    else:
        distancePredict = distancePredict + 'mi'
    querySearchPredict = request.form['querySearchPredict']
    topTweetsPredict = bool(int(request.form['topTweetsPredict']))
    maxTweetsPredict = int(request.form['maxTweetsPredict'])
    langPredict = request.form['langPredict']
    if (langPredict == 'None') or (langPredict == ''):
        langPredict = None 
    
    pred_results, pred_text = NB_analyzer.run_prediction(userNamePredict, geoLocationPredict, distancePredict, querySearchPredict,\
                           maxTweetsPredict, topTweetsPredict, langPredict, printAll=True)
    pred_results = pred_results.replace('\n', '<br/>')
    pred_text = pred_text.replace('\n', '<br/>')
        
    return render_template('nb_results.html', ticker=ticker, years=years, userName=userName,\
                           geoLocation=geoLocation, distance=distance, sinceDate=sinceDate, untilDate=untilDate,\
                               querySearch=querySearch, topTweets=topTweets, maxTweets=maxTweets, lang=lang,\
                                   deltaInterval=deltaInterval, changeThreshold=changeThreshold, trainSize=trainSize,\
                                       useIDF=useIDF, do_downsample=do_downsample, useStopwords=useStopwords,\
                                           addStopwords=addStopwords,\
                                           script_p=script_p, div_p=div_p,\
                                               count_report=count_report,\
                                                   report=report, most_inform=most_inform,\
                                                       confusion_matrix=conf_mat, recollectData=recollectData,\
                                                           recollectTweets=recollectTweets, userNamePredict=userNamePredict, geoLocationPredict=geoLocationPredict,\
                                                               distancePredict=distancePredict, querySearchPredict=querySearchPredict,\
                                                                   topTweetsPredict=topTweetsPredict, maxTweetsPredict=maxTweetsPredict, langPredict=langPredict,\
                                                                   pred_results=pred_results, pred_text=pred_text)
        
@app.route('/define_dependency', methods=['GET', 'POST'])
def define_dependency_parameters():
    return render_template('dependency_results.html', analyzeTicker='DAL', metricTickers='NDAQ,XOM', years=3, recollectData=True,\
                           analyzeInterval=3, metricInterval=7, changeThreshold=0.02, trainSize=0.8, kFold=5, \
                               scaleSVM=True, c_svm=1, SVM_kernel='rbf', SVM_gamma='scale', \
                                   KNN_neighbors=10, KNN_weighting='distance', RF_n_estimators=100, RF_criterion='entropy')

@app.route('/dependency_results', methods=['GET', 'POST'])
def dependency_results():
    
    analyzeTicker = request.form['analyzeTicker']
    metricTickers = request.form['metricTickers']
    metricTickers_tokenized = metricTickers.split(',')
    years = float(request.form['years'])
    recollectData = bool(int(request.form['recollectData']))
    
    analyzeInterval = parse_input(request.form['analyzeInterval'], None, expect_output='int')
    metricInterval = parse_input(request.form['metricInterval'], None, expect_output='int')
    changeThreshold = parse_input(request.form['changeThreshold'], 0, expect_output='float')
    trainSize = parse_input(request.form['trainSize'], 0.8, expect_output='float')
    kFold = parse_input(request.form['kFold'], 5, expect_output='int')
    
    scaleSVM = bool(int(request.form['scaleSVM']))
    c_svm = parse_input(request.form['c_svm'], 1, expect_output='float')
    SVM_kernel = (request.form['SVM_kernel']).split(',')
    SVM_degree = parse_input(request.form['SVM_degree'], 3, expect_output='int')
    coeff_svm = parse_input(request.form['coeff_svm'], 0, expect_output='float')
    SVM_gamma = (request.form['SVM_gamma']).split(',')
    SVM_gamma_temp = []
    for i_gamma in SVM_gamma:
        try:
            SVM_gamma_temp.append(float(i_gamma))
        except:
            SVM_gamma_temp.append(i_gamma)
    SVM_gamma = SVM_gamma_temp
    
    KNN_neighbors = parse_input(request.form['KNN_neighbors'], 4, expect_output='int')
    KNN_weighting = (request.form['KNN_weighting']).split(',')
    
    RF_n_estimators = parse_input(request.form['RF_n_estimators'], 100, expect_output='int')
    RF_criterion = [request.form['RF_criterion']]
    if RF_criterion[0] == 'both':
        RF_criterion = ['gini','entropy']
    
    SVM_grid = {'C': c_svm, 'kernel': SVM_kernel, 'degree': SVM_degree, 'gamma': SVM_gamma, 'coef0': coeff_svm}
    KNN_grid = {'n_neighbors': KNN_neighbors, 'weights': KNN_weighting}
    RF_grid = {'n_estimators': RF_n_estimators, 'criterion': RF_criterion}
    
    dependency_analyzer = Stock_Dependency_Analyzer()
    if recollectData:
        dependency_analyzer.collect_data(analyzeTicker, metricTickers_tokenized, years)
        analyzerData = dependency_analyzer.analyzerStockData_
        metricData = dependency_analyzer.metricStockData_
        sqlite_connection = engine.connect()
        analyzerData.to_sql(name='AnalyzerData', con=sqlite_connection, if_exists='replace')
        for idx, df in enumerate(metricData):
            df.to_sql(name='MetricData'+str(idx), con=sqlite_connection, if_exists='replace')
        sqlite_connection.close()
    else:
        sqlite_connection = engine.connect()
        analyzerData = pd.read_sql('select * from AnalyzerData', sqlite_connection)
        metricData = [pd.read_sql('select * from MetricData' + str(idx), sqlite_connection) \
                      for idx in range(len(metricTickers_tokenized))]
        sqlite_connection.close()
        dependency_analyzer.load_data(analyzerData, metricData, analyzeTicker, metricTickers_tokenized)
    
    p, ps, scores_list, report_list, conf_mat_list, svm_params, knn_params, rf_params, predictionDF,\
        analyzeInterval, metricInterval, changeThreshold, trainSize, kFold = Stock_Dependency_Grid_Search(dependency_analyzer, trainSize, kFold, analyzeInterval, metricInterval, changeThreshold,\
                                 scaleSVM, SVM_grid, KNN_grid, RF_grid, doHTML=True)

    script_p, div_p = components(p)
    report_list = [report.to_html(index=False) for report in report_list]
    conf_mat_list = [conf_mat.to_html(index=False, bold_rows=True) for conf_mat in conf_mat_list]
    predictionDF = predictionDF.to_html(index=False)
    
    c_svm = svm_params['C']
    SVM_kernel = svm_params['kernel']
    SVM_degree = svm_params['degree']
    coeff_svm = svm_params['coef0']
    SVM_gamma = svm_params['gamma']
    
    KNN_neighbors = knn_params['n_neighbors']
    KNN_weighting = knn_params['weights']
    
    RF_n_estimators = rf_params['n_estimators']
    RF_criterion = rf_params['criterion']
    
    return render_template('dependency_results.html', analyzeTicker=analyzeTicker, metricTickers=metricTickers, years=years, recollectData=recollectData,\
                           analyzeInterval=analyzeInterval, metricInterval=metricInterval, changeThreshold=changeThreshold,\
                               trainSize=trainSize, kFold=kFold, scaleSVM=scaleSVM, c_svm=c_svm, SVM_kernel=SVM_kernel, SVM_degree=SVM_degree, coeff_svm=coeff_svm, SVM_gamma=SVM_gamma,\
                                   KNN_neighbors=KNN_neighbors, KNN_weighting=KNN_weighting, RF_n_estimators=RF_n_estimators,\
                                       RF_criterion=RF_criterion, script_p=script_p, div_p=div_p, scatter_url=ps,\
                                           report_svm=report_list[0], conf_mat_svm=conf_mat_list[0], scores_svm=scores_list[0],\
                                               report_knn=report_list[1], conf_mat_knn=conf_mat_list[1], scores_knn=scores_list[1],\
                                                   report_rf=report_list[2], conf_mat_rf=conf_mat_list[2], scores_rf=scores_list[2],\
                                                       pred_df=predictionDF)

@app.route('/define_timing', methods=['GET', 'POST'])
def define_timing_parameters():
    return render_template('timing_results.html', ticker='QQQ', years=1, neg_threshold='0:0.005:0.02', pos_threshold='0:0.005:0.02', \
                           sell_adj='0:0.25:1', buy_adj='0:0.25:1')

@app.route('/timing_results', methods=['GET', 'POST'])
def timing_results():
    
    ticker = request.form['ticker']
    years = float(request.form['years'])
    
    neg_thresh = parse_input(request.form['neg_threshold'], 0, expect_output='float')
    pos_thresh = parse_input(request.form['pos_threshold'], 0, expect_output='float')
    sell_adj = parse_input(request.form['sell_adj'], 0, expect_output='float')
    buy_adj = parse_input(request.form['buy_adj'], 0, expect_output='float')
    
    timing_analyzer = Stock_Timing_Analyzer()
    timing_analyzer.collect_data(ticker, years)
    stats_df, results_df, p_hist, p, best_neg_thresh, best_pos_thresh, best_sell_adj, best_buy_adj \
                = Stock_Timing_Grid_Search(timing_analyzer, neg_thresh, pos_thresh,  sell_adj, buy_adj, doHTML=True)
    
    script_p_hist, div_p_hist = components(p_hist)
    script_p, div_p = components(p)
    
    stats_df = stats_df.to_html()
    results_df = results_df.to_html(index=False)
    
    return render_template('timing_results.html', ticker=ticker, years=years, neg_threshold=best_neg_thresh, pos_threshold=best_pos_thresh, \
                           sell_adj=best_sell_adj, buy_adj=best_buy_adj, stats_df=stats_df, results_df=results_df, \
                               script_p=script_p, div_p=div_p, script_p_hist=script_p_hist, div_p_hist=div_p_hist)