# -*- coding: utf-8 -*-
"""
Created on Sat May 23 19:20:16 2020

@author: timot
"""


from flask import Flask, render_template, request
from bokeh.embed import components
from Stock_NB_Analyzer import Stock_NB_Analyzer
from Stock_Dependency_Analyzer import Stock_Dependency_Analyzer
from nltk.corpus import stopwords
from sqlalchemy import create_engine
import pandas as pd

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
                                   deltaInterval=3, trainSize=0.8,\
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
    deltaInterval = int(request.form['deltaInterval'])
    trainSize = float(request.form['trainSize'])
    useIDF = bool(int(request.form['useIDF']))
    do_downsample = bool(int(request.form['do_downsample']))
    useStopwords = bool(int(request.form['useStopwords']))
    if useStopwords:
        stopwordsList = stopwords.words('english')
    else:
        stopwordsList = []
    addStopwords = request.form['addStopwords']
    add_stopwords_tokenized = addStopwords.split(',')
    if add_stopwords_tokenized:
        for word in add_stopwords_tokenized:
            stopwordsList.append(word)
        stopwordsList = list(set(stopwordsList))
    if not stopwordsList:
        stopwordsList = None
        
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
    
    count_report = NB_analyzer.correlate_tweets(deltaInterval)
    count_report = count_report.replace('\n', '<br/>')
    p = NB_analyzer.plot_data(deltaInterval, isBokeh=True)
    script_p, div_p = components(p)
    report, most_inform, conf_mat = NB_analyzer.create_classifier(trainSize, stopwordsList, useIDF, do_downsample, do_stat=True, numFeatures=10, doHTML=True)
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
                                   deltaInterval=deltaInterval, trainSize=trainSize,\
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
    return render_template('dependency_results.html', analyzeTicker='TSLA', metricTickers='GOLD,XOM', years=1, recollectData=True,\
                           analyzeInterval=3, metricInterval=7, trainSize=0.8, SVM_gamma='scale', KNN_neighbors=3, KNN_weighting='uniform')

@app.route('/dependency_results', methods=['GET', 'POST'])
def dependency_results():
    
    analyzeTicker = request.form['analyzeTicker']
    metricTickers = request.form['metricTickers']
    metricTickers_tokenized = metricTickers.split(',')
    years = float(request.form['years'])
    recollectData = bool(int(request.form['recollectData']))
    analyzeInterval = int(request.form['analyzeInterval'])
    metricInterval = int(request.form['metricInterval'])
    changeThreshold = request.form['changeThreshold']
    if changeThreshold == '':
        changeThreshold = 0
    else:
        changeThreshold = float(changeThreshold)
    trainSize = float(request.form['trainSize'])
    SVM_kernel = request.form['SVM_kernel']
    SVM_degree = request.form['SVM_degree']
    if SVM_degree == '':
        SVM_degree = 3
    else:
        SVM_degree = int(SVM_degree)
    SVM_gamma = request.form['SVM_gamma']
    KNN_neighbors = int(request.form['KNN_neighbors'])
    KNN_weighting = request.form['KNN_weighting']
    
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
    # dependency_analyzer.collect_data(analyzeTicker, metricTickers_tokenized, years)
    p, ps = dependency_analyzer.build_correlation(analyzeInterval, metricInterval, changeFilter=changeThreshold, doHTML=True)
    script_p, div_p = components(p)
    report_list, conf_mat_list = dependency_analyzer.create_all_classifiers(SVM_kernel, SVM_degree, SVM_gamma, \
                                               KNN_neighbors, KNN_weighting, \
                                                   trainSize=trainSize, doHTML=True)
    report_list = [report.to_html(index=False) for report in report_list]
    conf_mat_list = [conf_mat.to_html(index=False, bold_rows=True) for conf_mat in conf_mat_list]
    predictionDF = dependency_analyzer.run_prediction()
    predictionDF = predictionDF.to_html(index=False)
    
    return render_template('dependency_results.html', analyzeTicker=analyzeTicker, metricTickers=metricTickers, years=years, recollectData=recollectData,\
                           analyzeInterval=analyzeInterval, metricInterval=metricInterval, changeThreshold=changeThreshold,\
                               trainSize=trainSize, SVM_kernel=SVM_kernel, SVM_degree=SVM_degree, SVM_gamma=SVM_gamma,\
                                   KNN_neighbors=KNN_neighbors, KNN_weighting=KNN_weighting, script_p=script_p, div_p=div_p, scatter_url=ps,\
                                       report_lr=report_list[0], conf_mat_lr=conf_mat_list[0],\
                                           report_svm=report_list[1], conf_mat_svm=conf_mat_list[1],\
                                               report_knn=report_list[2], conf_mat_knn=conf_mat_list[2], pred_df=predictionDF)
        