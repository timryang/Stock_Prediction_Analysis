# -*- coding: utf-8 -*-
"""
Created on Sat May 23 19:20:16 2020

@author: timot
"""


from flask import Flask, render_template, request
from bokeh.embed import components
from Stock_NB_Analyzer import Stock_NB_Analyzer
from Stock_Dependency_Analyzer import Stock_Dependency_Grid_Search
from nltk.corpus import stopwords
from sqlalchemy import create_engine
import pandas as pd
import numpy as np

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
    deltaInterval = int(request.form['deltaInterval'])
    changeThreshold = request.form['changeThreshold']
    if changeThreshold == '':
        changeThreshold = 0
    else:
        changeThreshold = float(changeThreshold)
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
    
    count_report = NB_analyzer.correlate_tweets(deltaInterval, changeThreshold)
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
    return render_template('dependency_results.html', analyzeTicker='DAL', metricTickers='NDAQ,XOM', years=3,\
                           analyzeInterval=3, metricInterval=7, changeThreshold=0.02, trainSize=0.8, kFold=5, \
                               scaleSVM=True, c_svm=1, SVM_kernel='rbf', SVM_gamma='scale', \
                                   KNN_neighbors=10, KNN_weighting='distance', RF_n_estimators=100, RF_criterion='entropy')

@app.route('/dependency_results', methods=['GET', 'POST'])
def dependency_results():
    
    analyzeTicker = request.form['analyzeTicker']
    metricTickers = request.form['metricTickers']
    metricTickers_tokenized = metricTickers.split(',')
    years = float(request.form['years'])
    
    analyzeInterval = request.form['analyzeInterval']
    if ':' in analyzeInterval:
        range_vals = [int(val) for val in analyzeInterval.split(':')]
        analyzeInterval = list(np.arange(range_vals[0], range_vals[2], range_vals[1]))
    else:
        analyzeInterval = [int(val) for val in analyzeInterval.split(',')]
    metricInterval = request.form['metricInterval']
    if ':' in metricInterval:
        range_vals = [int(val) for val in metricInterval.split(':')]
        metricInterval = list(np.arange(range_vals[0], range_vals[2], range_vals[1]))
    else:
        metricInterval = [int(val) for val in metricInterval.split(',')]
    changeThreshold = request.form['changeThreshold']
    if changeThreshold == '':
        changeThreshold = [0]
    elif ':' in changeThreshold:
        range_vals = [float(val) for val in changeThreshold.split(':')]
        changeThreshold = list(np.arange(range_vals[0], range_vals[2], range_vals[1]))
    else:
        changeThreshold = [float(val) for val in changeThreshold.split(',')]
    trainSize = request.form['trainSize']
    if ':' in trainSize:
        range_vals = [float(val) for val in trainSize.split(':')]
        trainSize = list(np.arange(range_vals[0], range_vals[2], range_vals[1]))
    else:
        trainSize = [float(val) for val in trainSize.split(',')]
    kFold = request.form['kFold']
    if ':' in kFold:
        range_vals = [int(val) for val in kFold.split(':')]
        kFold = list(np.arange(range_vals[0], range_vals[2], range_vals[1]))
    else:
        kFold = [int(val) for val in kFold.split(',')]
    
    scaleSVM = bool(int(request.form['scaleSVM']))
    c_svm = request.form['c_svm']
    if ':' in c_svm:
        range_vals = [float(val) for val in c_svm.split(':')]
        c_svm = list(np.arange(range_vals[0], range_vals[2], range_vals[1]))
    else:
        c_svm = [float(val) for val in c_svm.split(',')]
    SVM_kernel = (request.form['SVM_kernel']).split(',')
    SVM_degree = request.form['SVM_degree']
    if SVM_degree == '':
        SVM_degree = [3]
    elif ':' in SVM_degree:
        range_vals = [int(val) for val in SVM_degree.split(':')]
        SVM_degree = list(np.arange(range_vals[0], range_vals[2], range_vals[1]))
    else:
        SVM_degree = [int(val) for val in SVM_degree.split(',')]
    coeff_svm = request.form['coeff_svm']
    if coeff_svm == '':
        coeff_svm = [0]
    elif ':' in coeff_svm:
        range_vals = [float(val) for val in coeff_svm.split(':')]
        coeff_svm = list(np.arange(range_vals[0], range_vals[2], range_vals[1]))
    else:
        coeff_svm = [float(val) for val in coeff_svm.split(',')]
    SVM_gamma = (request.form['SVM_gamma']).split(',')
    SVM_gamma_temp = []
    for i_gamma in SVM_gamma:
        try:
            SVM_gamma_temp.append(float(i_gamma))
        except:
            SVM_gamma_temp.append(i_gamma)
    SVM_gamma = SVM_gamma_temp
    
    KNN_neighbors = request.form['KNN_neighbors']
    if ':' in KNN_neighbors:
        range_vals = [int(val) for val in KNN_neighbors.split(':')]
        KNN_neighbors = list(np.arange(range_vals[0], range_vals[2], range_vals[1]))
    else:
        KNN_neighbors = [int(val) for val in KNN_neighbors.split(',')]
    KNN_weighting = (request.form['KNN_weighting']).split(',')
    
    RF_n_estimators = request.form['RF_n_estimators']
    if ':' in RF_n_estimators:
        range_vals = [int(val) for val in RF_n_estimators.split(':')]
        RF_n_estimators = list(np.arange(range_vals[0], range_vals[2], range_vals[1]))
    else:
        RF_n_estimators = [int(val) for val in RF_n_estimators.split(',')]
    RF_criterion = (request.form['RF_criterion']).split(',')
    
    SVM_grid = {'C': c_svm, 'kernel': SVM_kernel, 'degree': SVM_degree, 'gamma': SVM_gamma, 'coef0': coeff_svm}
    KNN_grid = {'n_neighbors': KNN_neighbors, 'weights': KNN_weighting}
    RF_grid = {'n_estimators': RF_n_estimators, 'criterion': RF_criterion}
    
    p, ps, scores_list, report_list, conf_mat_list, svm_params, knn_params, rf_params, predictionDF,\
        analyzeInterval, metricInterval, changeThreshold, trainSize, kFold = Stock_Dependency_Grid_Search(analyzeTicker, metricTickers_tokenized, years, trainSize, kFold, analyzeInterval, metricInterval, changeThreshold,\
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
    
    return render_template('dependency_results.html', analyzeTicker=analyzeTicker, metricTickers=metricTickers, years=years,\
                           analyzeInterval=analyzeInterval, metricInterval=metricInterval, changeThreshold=changeThreshold,\
                               trainSize=trainSize, kFold=kFold, scaleSVM=scaleSVM, c_svm=c_svm, SVM_kernel=SVM_kernel, SVM_degree=SVM_degree, coeff_svm=coeff_svm, SVM_gamma=SVM_gamma,\
                                   KNN_neighbors=KNN_neighbors, KNN_weighting=KNN_weighting, RF_n_estimators=RF_n_estimators,\
                                       RF_criterion=RF_criterion, script_p=script_p, div_p=div_p, scatter_url=ps,\
                                           report_svm=report_list[0], conf_mat_svm=conf_mat_list[0], scores_svm=scores_list[0],\
                                               report_knn=report_list[1], conf_mat_knn=conf_mat_list[1], scores_knn=scores_list[1],\
                                                   report_rf=report_list[2], conf_mat_rf=conf_mat_list[2], scores_rf=scores_list[2],\
                                                       pred_df=predictionDF)
        